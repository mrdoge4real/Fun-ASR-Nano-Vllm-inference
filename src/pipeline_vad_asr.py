#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import torch

from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.auto.auto_model_vllm import AutoModelVLLM


def ms_to_samples(ms: float, sr: int = 16000) -> int:
    """毫秒转采样点数。"""
    return int(ms * sr / 1000)


def load_audio_16k(path: str) -> np.ndarray:
    """读取音频并重采样到 16kHz，返回 float32 单声道 numpy 数组。"""
    data = load_audio_text_image_video(path, fs=16000)
    arr = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    if arr.ndim > 1:  # 多声道取平均
        arr = arr.mean(axis=1)
    arr = arr.reshape(-1)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def run_vad(audio_path: str, vad_model_path: str, device: str,
            speech_noise_thres: float, max_end_silence_time: int):
    """FSMN-VAD 离线切分，返回 [(start_ms, end_ms), ...]。"""
    vad_model = AutoModel(model=vad_model_path, device=device,
                          disable_update=True, disable_pbar=True)
    res = vad_model.generate(
        input=audio_path,
        batch_size=1,
        disable_pbar=True,
        max_end_silence_time=max_end_silence_time,
        speech_noise_thres=speech_noise_thres,
    )
    segments = res[0].get("value", []) if res else []
    return segments


def run_diarization(seg_audio, seg_meta, spk_model, device="cuda:0",
                    merge_thr=0.78, oracle_num=None):
    """对分段音频做说话人聚类，返回与 seg_meta 对齐的说话人 id 列表。

    复用 FunASR 官方 campplus 分段-聚类 diarization 逻辑（serve_vllm.py 同款）：
    sv_chunk 分段 → CAMP++ 提取说话人特征 → ClusterBackend 聚类 →
    postprocess 平滑 → distribute_spk 按时间重叠把说话人分配到每个句子。

    Args:
        seg_audio: 每段音频 (float32 mono 16k numpy)。
        seg_meta: [(start_ms, end_ms, dur_ms), ...]。
        spk_model: funasr.AutoModel 加载的 CAMP++ 说话人特征模型。
        device: 聚类计算设备。
        merge_thr: 聚类余弦合并阈值。
        oracle_num: 先验说话人数（None=自动估计）。

    Returns:
        list[int]: 每段的说话人 id（长度与 seg_audio 相同）。
    """
    from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
    from funasr.models.campplus.cluster_backend import ClusterBackend

    # [[start_s, end_s, audio], ...]
    vad_segs = [[st / 1000.0, et / 1000.0, np.ascontiguousarray(a)]
                for (st, et, _), a in zip(seg_meta, seg_audio)]
    chunks = sv_chunk(vad_segs, fs=16000)
    if not chunks:
        return [0] * len(seg_audio)

    speech_list = [ch[2] for ch in chunks]
    spk_res = spk_model.generate(input=speech_list, cache={}, is_final=True)
    embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)  # (N, 192)

    cluster = ClusterBackend(merge_thr=merge_thr).to(device)
    labels = cluster(embs.cpu(), oracle_num=oracle_num)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    all_sorted = sorted(chunks, key=lambda x: x[0])
    sv_output = postprocess(all_sorted, None, labels, embs.cpu())

    sentences = [{"text": "", "start": st, "end": et} for st, et, _ in seg_meta]
    distribute_spk(sentences, sv_output)
    return [s["spk"] for s in sentences]


def main():
    parser = argparse.ArgumentParser(description="FSMN-VAD + Fun-ASR-Nano (vLLM) 长音频转写")
    parser.add_argument("--audio", required=True, help="输入音频文件路径")
    parser.add_argument("--model-dir", default=None,
                        help="Fun-ASR-Nano 模型目录，默认使用仓库 models/Fun-ASR-Nano-2512")
    parser.add_argument("--vad-model", default=None,
                        help="FSMN-VAD 模型目录，默认使用仓库 models/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    parser.add_argument("--device", default="cuda:0", help="VAD + audio 编码器所在设备")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM 张量并行卡数")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    # VAD 参数
    parser.add_argument("--speech-noise-thres", type=float, default=0.7)
    parser.add_argument("--max-end-silence-time", type=int, default=600)
    parser.add_argument("--min-segment-ms", type=int, default=800,
                        help="短于该时长的语音段跳过，避免噪声幻听")
    parser.add_argument("--max-segment-ms", type=int, default=20000,
                        help="超长段二次切分阈值（含 padding），避免 Nano 在长段上退化；0 表示不切")
    parser.add_argument("--pad-ms", type=int, default=200,
                        help="每段前后各外扩的时长，避免切掉首尾字")
    # ASR 参数
    parser.add_argument("--max-new-tokens", type=int, default=256, help="每段最大生成 token 数")
    parser.add_argument("--language", default="中文")
    parser.add_argument("--no-itn", action="store_true", help="关闭文本规整")
    parser.add_argument("--output", default=None, help="可选：结果保存为 JSONL 文件")
    # 说话人分离
    parser.add_argument("--spk-model", default=None,
                        help="说话人分离模型目录，默认使用仓库 models/speech_campplus_sv_zh_en_16k-common_advanced")
    parser.add_argument("--no-spk", action="store_true", help="关闭说话人分离")
    parser.add_argument("--spk-num", type=int, default=None, help="先验说话人数（None=自动估计）")
    args = parser.parse_args()

    model_dir = args.model_dir or os.path.join(
        "models", "Fun-ASR-Nano-2512")
    vad_model_dir = args.vad_model or os.path.join(
        "models", "speech_fsmn_vad_zh-cn-16k-common-pytorch")

    print("=" * 70)
    print("FSMN-VAD + Fun-ASR-Nano (vLLM) 长音频转写流水线")
    print("=" * 70)
    print(f"  音频: {args.audio}")
    print(f"  VAD: speech_noise_thres={args.speech_noise_thres}, "
          f"max_end_silence_time={args.max_end_silence_time}ms")
    print(f"  ASR: device={args.device}, tensor_parallel={args.tensor_parallel_size}, "
          f"dtype={args.dtype}")
    print()

    # 1. 加载 16k 音频（用于按 VAD 边界切片）
    t0 = time.perf_counter()
    audio = load_audio_16k(args.audio)
    sr = 16000
    total_ms = len(audio) / sr * 1000
    print(f"音频时长: {total_ms / 1000:.2f}s, 采样点: {len(audio)}")
    print(f"加载音频耗时: {time.perf_counter() - t0:.2f}s\n")

    # 2. FSMN-VAD 切分
    t0 = time.perf_counter()
    segments = run_vad(args.audio, vad_model_dir, args.device,
                       args.speech_noise_thres, args.max_end_silence_time)
    print(f"VAD 检测到 {len(segments)} 段语音 ({time.perf_counter() - t0:.2f}s)\n")

    if not segments:
        print("未检测到语音。")
        return

    # 3. 过滤 + 外扩 padding + 超长段二次切分
    pad_samples = ms_to_samples(args.pad_ms)
    max_seg_samples = ms_to_samples(args.max_segment_ms) if args.max_segment_ms > 0 else 0
    seg_audio, seg_meta = [], []
    for start_ms, end_ms in segments:
        dur = end_ms - start_ms
        if dur < args.min_segment_ms:
            continue
        s = max(0, ms_to_samples(start_ms) - pad_samples)
        e = min(len(audio), ms_to_samples(end_ms) + pad_samples)
        # 超长段按固定窗口二次切分，避免 Nano 在长段上退化
        if max_seg_samples and (e - s) > max_seg_samples:
            for ss in range(s, e, max_seg_samples):
                ee = min(e, ss + max_seg_samples)
                if ee - ss < ms_to_samples(args.min_segment_ms):
                    continue
                seg_audio.append(audio[ss:ee])
                seg_meta.append((ss / sr * 1000, ee / sr * 1000, (ee - ss) / sr * 1000))
        else:
            seg_audio.append(audio[s:e])
            seg_meta.append((start_ms, end_ms, dur))

    if not seg_audio:
        print("过滤后无有效语音段。")
        return

    # 4. 加载 Fun-ASR-Nano (vLLM) 并逐段转写
    t0 = time.perf_counter()
    engine = AutoModelVLLM(
        model=model_dir,
        hub="ms",
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
    )
    # 说话人分离模型（CAMP++；可用 --no-spk 关闭）
    spk_model = None
    if not args.no_spk:
        spk_model_dir = args.spk_model or os.path.join(
            "models", "speech_campplus_sv_zh_en_16k-common_advanced")
        spk_model = AutoModel(model=spk_model_dir, device=args.device,
                              disable_update=True, disable_pbar=True)
    print(f"模型加载耗时: {time.perf_counter() - t0:.1f}s\n")

    t0 = time.perf_counter()
    results = engine.generate(
        inputs=seg_audio,
        language=args.language,
        itn=not args.no_itn,
        max_new_tokens=args.max_new_tokens,
    )
    infer_time = time.perf_counter() - t0

    # 5. 组装每段结果
    full_text_parts, out_records = [], []
    for i, (r, (start_ms, end_ms, dur)) in enumerate(zip(results, seg_meta)):
        text = r["text"].strip()
        if text:
            full_text_parts.append(text)
        out_records.append({"index": i + 1, "start_ms": start_ms, "end_ms": end_ms,
                            "duration_ms": dur, "text": text})

    # 6. 说话人分离（可选）
    if spk_model is not None:
        t0 = time.perf_counter()
        spk_ids = run_diarization(seg_audio, seg_meta, spk_model,
                                  device=args.device, oracle_num=args.spk_num)
        for rec, spk in zip(out_records, spk_ids):
            rec["spk"] = int(spk)
        print(f"说话人分离耗时: {time.perf_counter() - t0:.2f}s\n")

    # 7. 输出
    print("=" * 70)
    print(f"转写结果 ({len(out_records)} 段, 推理 {infer_time:.2f}s)")
    print("=" * 70)
    for rec in out_records:
        ts = f"[{rec['start_ms'] / 1000:7.2f}s - {rec['end_ms'] / 1000:7.2f}s] ({rec['duration_ms']:5.0f}ms)"
        spk_str = f" [SPK{rec['spk']}]" if "spk" in rec else ""
        print(f"  段{rec['index']}: {ts}{spk_str}  {rec['text']}")

    full_text = "".join(full_text_parts)
    print("\n" + "-" * 70)
    print("拼接全文（逐句 + 说话人）:")
    for rec in out_records:
        spk = rec.get("spk")
        if spk is not None:
            print(f"SPK{spk}: {rec['text']}")
        else:
            print(f"      {rec['text']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.write(json.dumps({"full_text": full_text}, ensure_ascii=False) + "\n")
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
