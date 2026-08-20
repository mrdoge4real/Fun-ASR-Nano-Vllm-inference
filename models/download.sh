#!/usr/bin/env bash
# =============================================================================
# 从 ModelScope 下载 Fun-ASR-Nano vLLM 推理所需模型
#
# 下载三个模型：
#   1. Fun-ASR-Nano-2512                    —— ASR 模型（含音频编码器 + LLM）
#   2. speech_fsmn_vad_zh-cn-16k-common     —— FSMN-VAD 语音活动检测模型
#   3. speech_campplus_sv_zh_en_16k-common_advanced —— CAMP++ 说话人特征模型（说话人分离用）
#
# 用法：
#   bash models/download.sh
#   （脚本会把模型下载到本文件所在目录 models/ 下）
#
# 下载后的目录结构：
#   models/
#   ├── download.sh
#   ├── Fun-ASR-Nano-2512/
#   ├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
#   └── speech_campplus_sv_zh_en_16k-common_advanced/
# =============================================================================
set -e

# 切到脚本所在目录，保证 --local_dir 落在 models/ 下
cd "$(dirname "$(readlink -f "$0")")"

echo "==> [1/3] 下载 Fun-ASR-Nano-2512 (ASR 模型) ..."
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --local_dir Fun-ASR-Nano-2512

echo "==> [2/3] 下载 speech_fsmn_vad_zh-cn-16k-common-pytorch (VAD 模型) ..."
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
    --local_dir speech_fsmn_vad_zh-cn-16k-common-pytorch

echo "==> [3/3] 下载 speech_campplus_sv_zh_en_16k-common_advanced (CAMP++ 说话人特征模型) ..."
modelscope download --model iic/speech_campplus_sv_zh_en_16k-common_advanced \
    --local_dir speech_campplus_sv_zh_en_16k-common_advanced

echo "==> 下载完成。"
echo "    models/Fun-ASR-Nano-2512"
echo "    models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
echo "    models/speech_campplus_sv_zh_en_16k-common_advanced"
