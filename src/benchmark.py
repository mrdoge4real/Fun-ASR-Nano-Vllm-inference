#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Fun-ASR-Nano vLLM 服务并发压测脚本。

用 N 个并发请求同时把音频 POST 到服务端的 HTTP /asr 接口，
统计总耗时、成功率、吞吐(req/s)、延迟分布、RTF。

用法（先起服务：bash scripts/serve_vllm.sh，默认端口 10096）：
    python src/benchmark.py \
        --url http://localhost:10096/asr \
        --audio data/2speakers_example.wav \
        --num-requests 20 --concurrency 20 \
        --max-seconds 30
"""

import argparse
import concurrent.futures
import io
import json
import os
import time

import requests


def prepare_audio_bytes(audio_path: str, max_seconds=None):
    """读取音频；若指定 max_seconds 则截取前 N 秒后重新编码为 wav。

    Returns:
        (bytes, 文件名)
    """
    if max_seconds is None:
        with open(audio_path, "rb") as f:
            return f.read(), os.path.basename(audio_path)

    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) > int(max_seconds * sr):
        data = data[: int(max_seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="wav")
    return buf.getvalue(), os.path.basename(audio_path)


def transcribe_one(url, audio_bytes, filename, language, spk, timeout):
    """发一个 /asr 请求，返回 (是否成功, 耗时, 结果信息)。"""
    t0 = time.perf_counter()
    try:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {}
        if language:
            data["language"] = language
        if spk:
            data["spk"] = "true"
        resp = requests.post(url, files=files, data=data, timeout=timeout)
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            result = resp.json()
            return {
                "ok": True, "latency": elapsed, "status": 200,
                "text_len": len(result.get("text", "")),
                "audio_dur": result.get("duration", 0),
                "rtf": result.get("rtf", None),
            }
        return {"ok": False, "latency": elapsed, "status": resp.status_code,
                "error": resp.text[:200]}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"ok": False, "latency": elapsed, "status": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano 服务并发压测")
    parser.add_argument("--url", default="http://localhost:10096/asr",
                        help="服务端 /asr 接口地址")
    parser.add_argument("--audio", default="data/2speakers_example.wav",
                        help="压测用的音频文件")
    parser.add_argument("--num-requests", type=int, default=20, help="总请求数")
    parser.add_argument("--concurrency", type=int, default=20, help="并发数")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="截取音频前 N 秒再发（加快压测；不填则用整段音频）")
    parser.add_argument("--language", default=None, help="语种提示（传给服务端）")
    parser.add_argument("--spk", action="store_true", help="开启说话人分离")
    parser.add_argument("--timeout", type=float, default=600, help="单请求超时(秒)")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"音频文件不存在: {args.audio}")
        return
    if args.concurrency <= 0:
        print("--concurrency 必须 > 0")
        return

    audio_bytes, filename = prepare_audio_bytes(args.audio, args.max_seconds)
    audio_dur = (args.max_seconds
                 if args.max_seconds is not None else None)

    dur_desc = f"{audio_dur}s" if audio_dur is not None else "整段"
    print("=" * 60)
    print("Fun-ASR-Nano 服务并发压测")
    print("=" * 60)
    print(f"  URL:         {args.url}")
    print(f"  音频:        {args.audio} ({filename}, {dur_desc})")
    print(f"  请求数:      {args.num_requests}")
    print(f"  并发数:      {args.concurrency}")
    if args.spk:
        print("  说话人分离: 开启")
    print()

    # 预探测一次，确认服务在线
    try:
        probe = requests.get(args.url.replace("/asr", "/docs"), timeout=5)
    except Exception:
        probe = None
    if probe is None:
        print(f"! 无法连接服务端 {args.url}，请先启动服务（bash scripts/serve_vllm.sh）")
        return
    print(f"服务端在线: {args.url}\n")

    t_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(transcribe_one, args.url, audio_bytes, filename,
                      args.language, args.spk, args.timeout)
            for _ in range(args.num_requests)
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            print(f"\r  进度: {i}/{args.num_requests}", end="", flush=True)
    print("\n")

    wall = time.perf_counter() - t_start
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    lat = sorted(r["latency"] for r in results)
    n = len(lat)

    print("=" * 60)
    print(f"压测结果 ({n} 请求, 并发 {args.concurrency})")
    print("=" * 60)
    print(f"总耗时:     {wall:.2f}s")
    print(f"成功/失败:  {len(ok)} / {len(fail)}")
    print(f"吞吐:       {n / wall:.2f} req/s")
    if n:
        p50 = lat[n // 2]
        print(f"延迟:       平均 {sum(lat) / n:.2f}s | 最小 {lat[0]:.2f}s "
              f"| 最大 {lat[-1]:.2f}s | P50 {p50:.2f}s")
    if ok:
        avg_lat = sum(r["latency"] for r in ok) / len(ok)
        rtf_vals = [r["rtf"] for r in ok if r.get("rtf")]
        if rtf_vals:
            print(f"RTF:        平均 {sum(rtf_vals) / len(rtf_vals):.3f} (服务端上报)")
        avg_text = sum(r["text_len"] for r in ok) / len(ok)
        print(f"平均转写字数: {avg_text:.0f} 字")
    if fail:
        print("\n失败样例:")
        for r in fail[:3]:
            print(f"  status={r['status']}: {r.get('error', '')[:150]}")

    # 进程退出码：全成功则 0
    raise SystemExit(0 if not fail else 1)


if __name__ == "__main__":
    main()
