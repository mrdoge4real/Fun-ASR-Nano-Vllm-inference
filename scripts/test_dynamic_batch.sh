#!/usr/bin/env bash
# 服务动态批处理 vs 串行时间对比：同一个音频发 10 次（并发走动态批处理合批）
# 先起服务：bash scripts/serve_vllm.sh（默认 http://localhost:10096）
python src/test_dynamic_batch.py \
    --url http://localhost:10096/asr \
    --audio data/2speakers_example.wav \
    --num-requests 10 \
    --max-seconds 10
