#!/usr/bin/env bash
# Fun-ASR-Nano vLLM 部署服务（websocket / HTTP / OpenAI，流式 VAD）
# 用法：从仓库根目录运行。GPU / TP 按需修改。
export CUDA_VISIBLE_DEVICES=0
python src/serve_vllm.py \
    --port 10096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --spk-model models/speech_campplus_sv_zh_en_16k-common_advanced \
    --device cuda:0
