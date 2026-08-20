#!/usr/bin/env bash
# Fun-ASR-Nano 服务并发压测：20 请求 20 并发
# 先起服务：bash scripts/serve_vllm.sh（默认端口 10096）
# 参数均可改：--audio / --num-requests / --concurrency / --max-seconds / --url
python src/benchmark.py \
    --url http://localhost:10096/asr \
    --audio data/2speakers_example.wav \
    --num-requests 20 \
    --concurrency 20 \
    --max-seconds 30
