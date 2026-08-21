#!/usr/bin/env python3
"""Fun-ASR-Nano vLLM Inference Server.

统一服务，三个接口：
- HTTP REST: POST /asr (文件上传)
- WebSocket: ws://host:port/ws (流式音频, 流式 VAD)
- OpenAI API: POST /v1/audio/transcriptions (Whisper 兼容)

所有接口共享同一个 vLLM 引擎 + 动态(流式) VAD + SPK + 时间戳。

用法（在仓库根目录下）：
    CUDA_VISIBLE_DEVICES=0 python src/serve_vllm.py --port 8000
    CUDA_VISIBLE_DEVICES=0 python src/serve_vllm.py --port 8000 --tensor-parallel-size 2

客户端：
    python src/client_python.py --server ws://localhost:8000/ws --file data/2speakers_example.wav
"""

import argparse
import asyncio
import io
import json
import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def truncate_repetition(text, min_repeat_len=3, max_repeats=3):
    """Detect and truncate repetitive patterns in ASR output."""
    if not text or len(text) < 20:
        return text
    n = len(text)
    for length in range(min_repeat_len, min(n // max_repeats, 30)):
        for start in range(n - length * max_repeats):
            chunk = text[start:start + length]
            if text[start:start + length * max_repeats] == chunk * max_repeats:
                return text[:start + length]
    return text



try:
    from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    raise ImportError("pip install fastapi uvicorn python-multipart")

from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD
from funasr import AutoModel


# ============================================================
# Global state
# ============================================================
_engine = None
_vad_model = None
_spk_model = None
_batcher = None
_args = None
# 这把锁只保护共享模型的"次要"调用（VAD / 说话人分离），它们便宜且不是瓶颈。
# vLLM 的 generate 走 _BatchASR 动态批处理（单调用方），不再用锁串行。
_model_lock = threading.Lock()


class _BatchASR:
    """vLLM 动态批处理器：把并发的 ASR 推理请求攒成一波，合并成一次
    _engine.generate 调用（vLLM 单次调用内部会跨输入批处理）。

    为什么既能并发又不崩：
    - 只有 flush 后台任务这一个调用方碰 _engine.generate，天然避开 vLLM 0.13
      同步 generate 的多线程竞态（serial_utils 的 aux_buffers 并发置 None 会崩）；
    - 不同请求的音频段在"一次" generate 里被 vLLM 合批解码 → 真正的跨请求并发。

    批量触发条件（满足其一即开批）：
    1. 攒够 max_batch（默认 8）个音频段 → 立即唤醒 flush，不等窗口；
    2. 自上一轮 flush 起超过 flush_interval（默认 100ms）还没攒够 → 直接开推。
    """

    def __init__(self, engine, flush_interval=0.1, max_batch=8, max_idle_cycles=10):
        self.engine = engine
        self.flush_interval = flush_interval
        self.max_batch = max_batch          # 单次 generate 最多合批的音频段数
        self.max_idle_cycles = max_idle_cycles
        self._pending = {}                  # gen_kwargs_key -> [(future, seg_audios), ...]
        self._total_pending = 0             # 待处理音频段总数
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()
        self._flusher = None

    @staticmethod
    def _key_of(gen_kwargs):
        return json.dumps(gen_kwargs, sort_keys=True, ensure_ascii=False)

    async def infer(self, seg_audios, gen_kwargs):
        """提交一批音频段，返回按 seg_audios 顺序的推理结果列表。"""
        fut = asyncio.get_running_loop().create_future()
        async with self._lock:
            self._pending.setdefault(self._key_of(gen_kwargs), []).append((fut, seg_audios))
            self._total_pending += len(seg_audios)
            if self._total_pending >= self.max_batch:
                self._wake.set()            # 攒够了，不等窗口，立刻 flush
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush_loop())
        return await fut

    async def _flush_loop(self):
        idle = 0
        while True:
            # 等 flush_interval；_wake 被置位（攒够 max_batch）则立即醒来
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.flush_interval)
                self._wake.clear()
            except asyncio.TimeoutError:
                pass
            async with self._lock:
                pending, self._pending = self._pending, {}
                self._total_pending = 0
            if not pending:
                idle += 1
                if idle >= self.max_idle_cycles:  # 空闲约 1s 就退出，等下次 submit 再起
                    return
                continue
            idle = 0
            for key, items in pending.items():
                gen_kwargs = json.loads(key)
                flat = [(fut, seg) for fut, segs in items for seg in segs]
                if not flat:
                    continue
                acc = {}
                for i in range(0, len(flat), self.max_batch):
                    chunk = flat[i:i + self.max_batch]
                    all_inputs = [seg for _, seg in chunk]
                    try:
                        results = await asyncio.to_thread(
                            self.engine.generate, inputs=all_inputs, **gen_kwargs)
                        for (fut, _), res in zip(chunk, results):
                            acc.setdefault(fut, []).append(res)
                    except Exception as e:
                        logger.error(f"Batch ASR failed: {e}", exc_info=True)
                        for fut, _ in chunk:
                            if not fut.done():
                                fut.set_exception(e)
                for fut, res_list in acc.items():
                    if not fut.done():
                        fut.set_result(res_list)


def prepare_audio_for_inference(audio_data, sr, target_sr=16000):
    """Return mono float32 audio at target_sr for ASR inference."""
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        channel_axis = -1 if audio_data.shape[-1] <= audio_data.shape[0] else 0
        audio_data = audio_data.mean(axis=channel_axis)

    if sr != target_sr:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio_data.astype(np.float32), sr


def load_engine(args):
    global _engine, _vad_model, _spk_model, _batcher, _args
    _args = args
    if _engine is None:
        logger.info(f"Loading vLLM engine: {args.model}")
        _engine = FunASRNanoVLLM.from_pretrained(
            model=args.model, hub=args.hub, device=args.device, dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        _batcher = _BatchASR(_engine)
        logger.info(f"Loading VAD: {args.vad_model}")
        _vad_model = AutoModel(model=args.vad_model, device=args.device, disable_update=True)
        if args.spk_model:
            logger.info(f"Loading SPK: {args.spk_model}")
            _spk_model = AutoModel(model=args.spk_model, device=args.device, disable_update=True)
        else:
            logger.info("SPK disabled")
        logger.info("All models ready!")


def _vad_and_extract(audio_data, sr=16000, use_vad=True):
    """VAD 切段并抽取分段音频（VAD 是共享模型，调用方需持 _model_lock）。

    Returns:
        (seg_audios, seg_times): 分段音频列表 + [(start_ms, end_ms), ...]
    """
    if use_vad and len(audio_data) > sr * 1:
        vad_res = _vad_model.generate(input=audio_data, fs=sr)
        segments = vad_res[0]["value"]
    else:
        segments = [[0, int(len(audio_data) * 1000 / sr)]]

    seg_audios, seg_times = [], []
    for seg in segments:
        s0 = int(seg[0] * sr / 1000)
        s1 = int(seg[1] * sr / 1000)
        seg_audio = audio_data[s0:s1]
        if len(seg_audio) > sr * 0.3:
            seg_audios.append(seg_audio)
            seg_times.append((seg[0], seg[1]))
    return seg_audios, seg_times


def _apply_spk(output_segments, audio_data, sr):
    """给 output_segments 加 speaker 标签（CAMP++ 是共享模型，调用方需持锁）。"""
    from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
    from funasr.models.campplus.cluster_backend import ClusterBackend

    vad_segs = [[st, et, audio_data[int(st * sr):int(et * sr)]]
                for st, et in [(s["start"], s["end"]) for s in output_segments]]
    chunks = sv_chunk(vad_segs)
    if not chunks:
        return
    speech_list = [ch[2] for ch in chunks]
    spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
    embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
    cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
    labels = cluster(embs.cpu(), oracle_num=None)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    all_sorted = sorted(chunks, key=lambda x: x[0])
    sv_output = postprocess(all_sorted, None, labels, embs.cpu())
    sentences = [{"text": s["text"], "start": int(s["start"] * 1000), "end": int(s["end"] * 1000)}
                 for s in output_segments]
    distribute_spk(sentences, sv_output)
    for i, s in enumerate(sentences):
        output_segments[i]["speaker"] = f"SPK{s.get('spk', 0)}"


def _build_output(results, seg_times, audio_data, sr, use_spk=False, use_timestamp=True):
    """把推理结果拼装成输出段 + 可选说话人分离（SPK 部分调用方需持锁）。"""
    output_segments = []
    full_text_parts = []
    for i, (r, (start_ms, end_ms)) in enumerate(zip(results, seg_times)):
        r["text"] = truncate_repetition(r["text"])
        seg_info = {"text": r["text"], "start": start_ms / 1000, "end": end_ms / 1000}
        if use_timestamp and "timestamps" in r:
            offset = start_ms / 1000
            seg_info["words"] = [
                {"word": ts["token"], "start": ts["start_time"] + offset, "end": ts["end_time"] + offset}
                for ts in r["timestamps"]
            ]
        output_segments.append(seg_info)
        full_text_parts.append(r["text"])
    if use_spk and _spk_model is not None:
        _apply_spk(output_segments, audio_data, sr)
    return {"text": " ".join(full_text_parts), "segments": output_segments,
            "duration": len(audio_data) / sr}


def _vad_and_extract_locked(audio_data, sr=16000, use_vad=True):
    with _model_lock:
        return _vad_and_extract(audio_data, sr, use_vad)


def _build_output_locked(results, seg_times, audio_data, sr, use_spk=False, use_timestamp=True):
    with _model_lock:
        return _build_output(results, seg_times, audio_data, sr, use_spk, use_timestamp)


def _run_spk_diarization(audio_buffer, sentences):
    """在子线程中做说话人聚类，原地给 sentences 列表加 spk 字段。"""
    from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
    from funasr.models.campplus.cluster_backend import ClusterBackend
    if not sentences or _spk_model is None:
        return
    try:
        with _model_lock:
            vad_segs = [[s["start"] / 1000, s["end"] / 1000,
                         audio_buffer[int(s["start"] * 16):int(s["end"] * 16)]]
                        for s in sentences]
            chunks = sv_chunk(vad_segs)
            if not chunks:
                return
            speech_list = [ch[2] for ch in chunks]
            spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
            embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
            cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
            labels = cluster(embs.cpu(), oracle_num=None)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            all_sorted = sorted(chunks, key=lambda x: x[0])
            sv_output = postprocess(all_sorted, None, labels, embs.cpu())
            spk_sents = [{"text": s["text"], "start": int(s["start"]), "end": int(s["end"])}
                         for s in sentences]
            distribute_spk(spk_sents, sv_output)
            for i, ss in enumerate(spk_sents):
                sentences[i]["spk"] = ss.get("spk", 0)
    except Exception as e:
        logger.warning(f"SPK failed: {e}")


def build_openai_verbose_json(result, language=None):
    """Build OpenAI-compatible verbose_json while preserving FunASR extensions."""
    segments = []
    for i, seg in enumerate(result["segments"]):
        item = {
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
        }
        if "speaker" in seg:
            item["speaker"] = seg["speaker"]
        segments.append(item)

    return {
        "task": "transcribe",
        "language": language or "zh",
        "duration": result["duration"],
        "text": result["text"],
        "segments": segments,
    }


# ============================================================
# FastAPI App
# ============================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    """应用生命周期：启动时加载模型，关闭时回收批处理器。"""
    load_engine(_args)
    yield
    if _batcher is not None and _batcher._flusher is not None and not _batcher._flusher.done():
        _batcher._flusher.cancel()


app = FastAPI(title="Fun-ASR-Nano vLLM Server", version="1.0", lifespan=lifespan)


# --- HTTP REST: POST /asr ---
@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
    """ASR with file upload. Returns text + segments + timestamps + speaker."""
    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))

    hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None

    t0 = time.perf_counter()
    # 重采样 + VAD 切段（线程池，VAD 共享模型加锁保护）
    audio_data, sr = await asyncio.to_thread(prepare_audio_for_inference, audio_data, sr)
    seg_audios, seg_times = await asyncio.to_thread(_vad_and_extract_locked, audio_data, sr, True)
    if not seg_audios:
        result = {"text": "", "segments": [], "duration": len(audio_data) / sr}
    else:
        # vLLM 推理走动态批处理：并发请求被合批成一次 generate（真并发）
        gen_kwargs = {"max_new_tokens": 500}
        if language:
            gen_kwargs["language"] = language
        if hw_list:
            gen_kwargs["hotwords"] = hw_list
        results = await _batcher.infer(seg_audios, gen_kwargs)
        result = await asyncio.to_thread(
            _build_output_locked, results, seg_times, audio_data, sr,
            use_spk=spk, use_timestamp=timestamp)
    t1 = time.perf_counter()

    result["processing_time"] = round(t1 - t0, 3)
    result["rtf"] = round((t1 - t0) / result["duration"], 4) if result["duration"] > 0 else 0
    return JSONResponse(content=result)


# --- OpenAI API: POST /v1/audio/transcriptions ---
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="fun-asr-nano"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: str = Form(default="word"),
    spk: bool = Form(default=False),
):
    """OpenAI Whisper-compatible transcription API (extended with spk support)."""
    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))

    use_ts = "word" in timestamp_granularities or "segment" in timestamp_granularities
    audio_data, sr = await asyncio.to_thread(prepare_audio_for_inference, audio_data, sr)
    seg_audios, seg_times = await asyncio.to_thread(_vad_and_extract_locked, audio_data, sr, True)
    if not seg_audios:
        result = {"text": "", "segments": [], "duration": len(audio_data) / sr}
    else:
        gen_kwargs = {"max_new_tokens": 500}
        if language:
            gen_kwargs["language"] = language
        results = await _batcher.infer(seg_audios, gen_kwargs)
        result = await asyncio.to_thread(
            _build_output_locked, results, seg_times, audio_data, sr,
            use_spk=spk, use_timestamp=use_ts)

    if response_format == "text":
        return JSONResponse(content=result["text"])
    elif response_format == "verbose_json":
        return JSONResponse(content=build_openai_verbose_json(result, language=language))
    else:
        return JSONResponse(content={"text": result["text"]})


# --- WebSocket: ws://host:port/ws ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Streaming WebSocket ASR with dynamic VAD + SPK."""
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    vad = DynamicStreamingVAD(_vad_model)
    audio_buffer = np.array([], dtype=np.float32)
    locked_sentences = []
    language = None
    hotwords = None
    use_spk = False
    is_active = False

    try:
        while True:
            message = await websocket.receive()

            # 新版 Starlette：客户端断开时 receive() 返回 {"type":"websocket.disconnect"}
            # 而非抛异常；若不显式 break，再次调用 receive() 会抛 RuntimeError。
            if message["type"] == "websocket.disconnect":
                logger.info("WebSocket disconnected")
                break

            if "text" in message:
                cmd = message["text"].strip()
                if cmd.upper() == "START":
                    vad.reset()
                    audio_buffer = np.array([], dtype=np.float32)
                    locked_sentences = []
                    is_active = True
                    await websocket.send_json({"event": "started"})
                elif cmd.upper().startswith("LANGUAGE:"):
                    language = cmd[9:].strip() or None
                    await websocket.send_json({"event": "language_set", "language": language})
                elif cmd.upper().startswith("HOTWORDS:"):
                    hotwords = [w.strip() for w in cmd[9:].split(",") if w.strip()]
                    await websocket.send_json({"event": "hotwords_set", "hotwords": hotwords})
                elif cmd.upper().startswith("SPK:"):
                    use_spk = cmd[4:].strip().lower() in ("true", "1", "on", "yes")
                    await websocket.send_json({"event": "spk_set", "spk": use_spk})
                elif cmd.upper() == "STOP":
                    if is_active and len(audio_buffer) > 0:
                        # Final: process remaining audio
                        final_segs = vad.finalize()
                        for seg in final_segs:
                            seg_audio = audio_buffer[int(seg[0]*16):int(seg[1]*16)]
                            if len(seg_audio) > 8000:
                                gen_kw = {"max_new_tokens": 500}
                                if language: gen_kw["language"] = language
                                if hotwords: gen_kw["hotwords"] = hotwords
                                res = await _batcher.infer([seg_audio], gen_kw)
                                if res[0]["text"].strip():
                                    locked_sentences.append({
                                        "text": res[0]["text"], "start": seg[0], "end": seg[1]
                                    })

                        # Handle ongoing speech
                        if vad.is_speaking:
                            end_ms = int(len(audio_buffer) * 1000 / 16000)
                            start_ms = int(vad.current_speech_start) if hasattr(vad, 'current_speech_start') and vad.current_speech_start else 0
                            seg_audio = audio_buffer[int(start_ms*16):]
                            if len(seg_audio) > 8000:
                                gen_kw = {"max_new_tokens": 500}
                                if language: gen_kw["language"] = language
                                if hotwords: gen_kw["hotwords"] = hotwords
                                res = await _batcher.infer([seg_audio], gen_kw)
                                if res[0]["text"].strip():
                                    locked_sentences.append({
                                        "text": res[0]["text"], "start": start_ms, "end": end_ms
                                    })

                        # SPK: run full clustering on all sentences (only if enabled)
                        if use_spk and locked_sentences and _spk_model is not None:
                            await asyncio.to_thread(_run_spk_diarization, audio_buffer, locked_sentences)

                        await websocket.send_json({
                            "sentences": locked_sentences,
                            "is_final": True,
                            "duration_ms": int(len(audio_buffer) * 1000 / 16000),
                        })
                        is_active = False
                    await websocket.send_json({"event": "stopped"})

            elif "bytes" in message and is_active:
                pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, pcm])

                # Feed VAD
                new_confirmed = vad.feed(torch.from_numpy(pcm).float())
                for seg in new_confirmed:
                    seg_audio = audio_buffer[int(seg[0]*16):int(seg[1]*16)]
                    if len(seg_audio) > 8000:
                        gen_kw = {"max_new_tokens": 500}
                        if language: gen_kw["language"] = language
                        if hotwords: gen_kw["hotwords"] = hotwords
                        res = await _batcher.infer([seg_audio], gen_kw)
                        if res[0]["text"].strip():
                            locked_sentences.append({
                                "text": res[0]["text"], "start": seg[0], "end": seg[1]
                            })

                # Send partial update
                await websocket.send_json({
                    "sentences": locked_sentences,
                    "is_final": False,
                    "duration_ms": int(len(audio_buffer) * 1000 / 16000),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano vLLM Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="models/Fun-ASR-Nano-2512",
                        help="Fun-ASR-Nano 模型名或本地目录（默认仓库 models/ 下本地模型）")
    parser.add_argument("--hub", type=str, default="ms")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM 张量并行卡数")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vad-model", type=str,
                        default="models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        help="VAD 模型名或本地目录（默认仓库 models/ 下本地模型）")
    parser.add_argument("--spk-model", type=str,
                        default="models/speech_campplus_sv_zh_en_16k-common_advanced",
                        help="说话人分离模型名或本地目录（默认仓库 models/ 下本地模型；留空禁用）")
    _args = parser.parse_args()

    # 模型加载交给 FastAPI lifespan 生命周期（app 启动时执行）
    uvicorn.run(app, host=_args.host, port=_args.port)
