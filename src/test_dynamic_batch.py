#!/usr/bin/env python3
"""测试 serve_vllm.py 服务的动态批处理 vs 串行（无批处理）时间对比。

先启动服务：
    bash scripts/serve_vllm.sh   # 默认 http://localhost:10096

对比 N 个音频文件两种提交方式的总耗时：
1. 串行：逐个 POST 到服务端 /asr（服务端每次单独 generate，无并发合批）
2. 并发：N 个同时 POST（服务端 _BatchASR 把同时到达的音频段动态合批）

用法：
    python src/test_dynamic_batch.py --url http://localhost:10096/asr --num-requests 10
"""

import argparse
import concurrent.futures
import io
import os
import time

import numpy as np
import requests
import soundfile as sf


def build_clips(audio_path, num, max_seconds=None, sr=16000):
    """加载同一个音频，返回 num 份相同的 wav 字节（可选截取前 max_seconds 秒）。"""
    data, orig_sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if orig_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr).astype(np.float32)
    if max_seconds and len(data) > int(max_seconds * sr):
        data = data[:int(max_seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="wav")
    return [buf.getvalue()] * num


def post_asr(url, wav_bytes, timeout=300):
    t0 = time.perf_counter()
    resp = requests.post(url, files={"file": ("clip.wav", wav_bytes, "audio/wav")}, timeout=timeout)
    return time.perf_counter() - t0, resp


def main():
    parser = argparse.ArgumentParser(description="服务动态批处理 vs 串行时间对比")
    parser.add_argument("--url", default="http://localhost:10096/asr", help="服务端 /asr 接口")
    parser.add_argument("--audio", default="data/2speakers_example.wav")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="截取音频前 N 秒再发（默认整段；为加快测试可设 10）")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"音频不存在: {args.audio}")
        return

    # 探活
    try:
        requests.get(args.url.replace("/asr", "/docs"), timeout=5)
    except Exception:
        print(f"! 服务未启动: {args.url}，请先 bash scripts/serve_vllm.sh")
        return

    clips = build_clips(args.audio, args.num_requests, args.max_seconds)
    clip_dur = len(clips[0]) / 2 / 16000
    trunc = f"（截取前 {args.max_seconds}s）" if args.max_seconds else "（整段）"
    print(f"同一个音频 × {len(clips)} 个请求，每个时长 {clip_dur:.1f}s {trunc}")

    # 预热一个请求
    print("预热 ...")
    post_asr(args.url, clips[0], args.timeout)
    print()

    # 1) 串行：逐个发
    t0 = time.perf_counter()
    serial_lat = []
    for clip in clips:
        lat, _ = post_asr(args.url, clip, args.timeout)
        serial_lat.append(lat)
    t_serial = time.perf_counter() - t0
    print(f"【串行】{len(clips)} 次请求，总耗时 {t_serial:.3f}s | "
          f"单次 min={min(serial_lat):.3f}s max={max(serial_lat):.3f}s 累计={sum(serial_lat):.3f}s")

    # 2) 并发：同时发（服务端 _BatchASR 动态批处理）
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_requests) as ex:
        futures = [ex.submit(post_asr, args.url, c, args.timeout) for c in clips]
        conc_lat = [f.result()[0] for f in concurrent.futures.as_completed(futures)]
    t_concurrent = time.perf_counter() - t0
    print(f"【并发/动态批处理】{len(clips)} 次同时请求，总耗时 {t_concurrent:.3f}s | "
          f"单次 min={min(conc_lat):.3f}s max={max(conc_lat):.3f}s")

    print()
    print("=" * 56)
    print(f"加速比（串行 / 并发批处理）: {t_serial / t_concurrent:.2f}x")
    print("=" * 56)
    print("说明：串行总耗时≈各请求耗时之和；并发时服务端 _BatchASR 把同时到达的")
    print("      音频段合批成一次 generate，总耗时≈批内最长音频耗时。")


if __name__ == "__main__":
    main()
