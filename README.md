
# Fun-ASR-Nano · vLLM 语音转写（ASR）推理

基于 **Fun-ASR-Nano**（音频编码器 + Qwen3-0.6B LLM）+ **vLLM** 后端的语音转写工程。
音频编码器 / 适配器在 PyTorch 中运行，LLM 解码在 vLLM 中运行（支持张量并行），
可对长音频做高吞吐转写，并可选**说话人分离（diarization）**。

本仓库提供**两种部署模式**，核心区别在 **VAD 切分方式**：

| 模式 | 入口 | VAD 方式 | 特点 / 适用场景 |
| --- | --- | --- | --- |
| **离线长音频流水线** | `src/pipeline_vad_asr.py` | **离线** FSMN-VAD 整段切分 | 一次性转写整段音频；可过滤短段、外扩、超长段二次切分，质量优先 |
| **WebSocket 流式服务** | `src/serve_vllm.py` | **流式** DynamicStreamingVAD | 边传音频边出实时分句；STOP 后输出最终结果，适合实时对话场景 |

---

## 特性

- **vLLM 后端**：`enable_prompt_embeds=True`，LLM 解码走 vLLM，支持张量并行（多卡）
- **两种 VAD**：
  - 离线：FSMN-VAD 对整段音频切分，支持过滤短段、前后外扩 padding、超长段二次切分
  - 流式：DynamicStreamingVAD 边收音频边实时切分
- **说话人分离**：CAMP++（`speech_campplus_sv_zh_en_16k-common_advanced`）分段-聚类，给每句转写标 `SPK n`
- **三种服务接口**（流式服务内）：WebSocket / HTTP REST `POST /asr` / OpenAI Whisper 兼容接口
- **并发压测**：`src/benchmark.py` 一键测服务端并发能力（吞吐 / 延迟 / RTF）

---

## 仓库结构

```
Fun-ASR-Nano-Vllm-inference/
├── README.md                      # 本文档
├── requirements.txt               # 依赖（顶部内置 PyTorch cu129 索引）
├── src/                           # Python 脚本（顶层，无子包）
│   ├── pipeline_vad_asr.py        # 【离线】FSMN-VAD + Nano 长音频转写流水线
│   ├── serve_vllm.py              # 【流式】vLLM websocket / HTTP / OpenAI 服务
│   ├── client_python.py           # WebSocket 客户端（文件 / 麦克风）
│   └── benchmark.py               # 服务并发压测
├── scripts/                       # 启动脚本
│   ├── pipline_vad_asr.sh         # 离线流水线启动
│   ├── serve_vllm.sh              # 流式服务启动
│   ├── client_python.sh           # 客户端启动
│   └── benchmark.sh               # 并发压测启动
├── data/                          # 测试音频
└── models/                        # 模型（git 已忽略，只保留 download.sh）
    └── download.sh                # 一键下载 3 个模型
```

---

## 快速开始

```bash
# 1. 创建 conda 环境（Python 3.11）
conda create -n funasr-vllm python=3.11
conda activate funasr-vllm

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型（ASR + VAD + CAMP++ 说话人分离）
bash models/download.sh
```

> **关于 torch 的 `+cu129` 版本**：`requirements.txt` 顶部已加了
> `--extra-index-url https://download.pytorch.org/whl/cu129`。`torch==2.9.0+cu129`
> 这类带 CUDA 后缀的构建**只发布在 PyTorch 官方索引**上，普通 PyPI / 清华镜像没有，
> 所以必须带这个索引才能装。
>
> **需要 CUDA GPU**，驱动需支持 CUDA 12.9（驱动 ≥ 570.124）。

---

## 本机运行环境

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Linux（内核 6.8） |
| CPU | 2 × Intel Xeon Platinum 8358 @ 2.60GHz（2 路 × 32 核 × 2 线程 = 128 逻辑 CPU，2 个 NUMA 节点） |
| 内存 | 1.0 TiB |
| GPU | 8 × NVIDIA A800-SXM4-80GB（每卡 80GB HBM） |
| 显卡驱动 | 570.148.08（≥ 570.124，满足 CUDA 12.9 运行要求） |
| nvidia-smi 显示 CUDA | 12.8（驱动标注的最高版本） |
| PyTorch 构建 | `2.9.0+cu129`（torchaudio `2.9.0+cu129` / torchvision `0.24.0+cu129`） |
| Python | 3.11（conda 环境 `funasr-vllm`） |

> **CUDA 版本说明**：`nvidia-smi` 标的是驱动支持的最高 CUDA 版本（12.8）；
> 实际用的 PyTorch 是 **cu129** 构建，本机驱动已满足其运行要求，实测推理正常。

---

## 模型下载

`bash models/download.sh` 会把 3 个模型下载到 `models/`：

| 模型 | ModelScope ID | 作用 |
| --- | --- | --- |
| Fun-ASR-Nano-2512 | `FunAudioLLM/Fun-ASR-Nano-2512` | 端到端 ASR（音频编码器 + Qwen3-0.6B LLM） |
| FSMN-VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 语音活动检测（离线 + 流式通用） |
| CAMP++ SV | `iic/speech_campplus_sv_zh_en_16k-common_advanced` | 说话人特征提取（说话人分离用） |

> 注意：ModelScope 上的 `speech_campplus_speaker-diarization_common` 是**组合模型**，
> 仓库里不带权重，无法被 `AutoModel` 直接加载；实际用到的是它的子模型
> `speech_campplus_sv_zh_en_16k-common_advanced`（CAMP++ 说话人特征权重）。

---

## 模式一：离线长音频流水线（`pipeline_vad_asr.py`）

**离线 VAD**：先用 FSMN-VAD 对整段音频做一次切分，再做过滤/外扩/二次切分，
最后批量喂给 vLLM 转写。适合"给一整段音频，要一份高质量全文"的场景。

### 处理流程

```
长音频 (wav/mp3)
  → ① 加载并重采样 16k 单声道
  → ② FSMN-VAD 离线切分语音段
  → ③ 过滤短段 / 前后外扩 padding / 超长段二次切分
  → ④ AutoModelVLLM 逐段批量转写（vLLM）
  → ⑤ CAMP++ 说话人聚类（可选，默认开）
  → ⑥ 输出逐句 + SPK + 拼接全文
```

### 运行

```bash
bash scripts/pipline_vad_asr.sh
```

等价命令：

```bash
python src/pipeline_vad_asr.py \
    --audio data/2speakers_example.wav \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.3 \
    --spk-model models/speech_campplus_sv_zh_en_16k-common_advanced \
    --device cuda:0
```

### 输出示例

```
拼接全文（逐句 + 说话人）:
SPK0: 大家稍微安静一下，我们开始看上个月的销售复盘。
SPK1: 整体来看，华东和华南两个大区的业绩是达标了的。
SPK1: 尤其是华东，超额完成了百分之十五。
SPK0: 但是华北的表现非常不理想，整体销售额只有预估的百分之六十左右。
```

### 主要参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--speech-noise-thres` | `0.7` | VAD 语音/噪声判定阈值 |
| `--max-end-silence-time` | `600` | 段尾静音毫秒数，达到即断句 |
| `--min-segment-ms` | `800` | 短于此的语音段跳过（防噪声幻听） |
| `--max-segment-ms` | `20000` | 超长段二次切分阈值；`0` 不切 |
| `--pad-ms` | `200` | 每段前后外扩毫秒数（防切掉首尾字） |
| `--spk-model` | `models/speech_campplus_sv_zh_en_16k-common_advanced` | 说话人分离模型 |
| `--no-spk` | 关 | 关闭说话人分离 |
| `--spk-num` | 自动 | 先验说话人数（多人对话时可给准确值提升效果） |
| `--language` | `中文` | 语种提示 |
| `--no-itn` | 关 | 关闭文本规整 |
| `--output` | 无 | 结果存为 JSONL（含 `spk` 字段） |

---

## 模式二：WebSocket 流式服务（`serve_vllm.py`）

**流式 VAD**：DynamicStreamingVAD 边收音频边实时切分，客户端边传边收实时分句；
发送 `STOP` 后服务端处理剩余音频并返回带说话人的最终结果。

### 服务接口

| 接口 | 说明 |
| --- | --- |
| `ws://host:port/ws` | WebSocket 流式音频转写（主接口） |
| `POST /asr` | HTTP 文件上传转写（表单 `file`，可选 `language` / `spk`） |
| `POST /v1/audio/transcriptions` | OpenAI Whisper 兼容接口 |

### 启动服务

```bash
bash scripts/serve_vllm.sh
# 默认配置见脚本：GPU 3,5、TP=2、端口 10096、说话人分离开启
```

手动启动：

```bash
CUDA_VISIBLE_DEVICES=3,5 python src/serve_vllm.py \
    --port 10096 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.3 \
    --spk-model models/speech_campplus_sv_zh_en_16k-common_advanced \
    --device cuda:0
```

### WebSocket 协议

文本命令：`START` / `STOP` / `LANGUAGE:中文` / `HOTWORDS:张三,李四` / `SPK:true`。

音频数据：原始 **int16 PCM 16kHz 单声道** 二进制帧。

返回：命令回 `{"event": ...}`；音频期间实时回 `{"sentences": [...], "is_final": false}`；
`STOP` 后回 `{"sentences": [...], "is_final": true}`（每句含 `text` / `start` / `end` / `spk`）。

### 客户端

```bash
# 文件模式
bash scripts/client_python.sh
# 等价：python src/client_python.py --server ws://localhost:10096/ws --file data/2speakers_example.wav

# 麦克风模式（实时录制；需要 sounddevice：pip install sounddevice）
python src/client_python.py --server ws://localhost:10096/ws --mic

# 关闭说话人分离显示
python src/client_python.py --server ws://localhost:10096/ws --file audio.wav --no-spk
```

客户端按句显示（彩色 SPK）：

```
  [0.0s-4.6s]  SPK0 大家稍微安静一下，我们开始看上个月的销售复盘。
  [4.6s-10.3s] SPK1 整体来看，华东和华南两个大区的业绩是达标了的。
```

---

## 说话人分离（diarization）

说话人分离用的模型是 **CAM++**（FunASR 官方说话人识别模型，模型 ID
`speech_campplus_sv_zh_en_16k-common_advanced`；在 FunASR 标准 `AutoModel` 里可用
`spk_model="cam++"` 简写）。它输出 **192 维说话人特征**，再配合官方 `campplus`
组件的分段-聚类逻辑，把说话人标签贴到每个转写句子上。

```
sv_chunk 分段 → CAM++ 提取 192 维说话人特征 → ClusterBackend 聚类
→ postprocess 平滑 → distribute_spk 按时间重叠给每句标 SPK
```

两种模式默认开启：

- 离线模式：`--no-spk` 关闭，`--spk-num` 可给先验人数
- 流式模式：客户端发 `SPK:true` 请求；服务端 `--spk-model` 指定模型

> 注意：
> - 说话人分离需要整段音频才能聚类，所以流式模式下 SPK 只在 `STOP`
>   后的**最终结果**里出现，实时分句不带 spk
> - 音频太短（聚类段 < 20 个 chunk）时官方逻辑会全部标 `SPK 0`，属正常现象

---

## 并发压测（`benchmark.py`）

测服务端的并发能力（HTTP `/asr` 接口，共享 vLLM 引擎）：

```bash
bash scripts/benchmark.sh
# 20 请求、20 并发，每个请求发前 30 秒音频（可改）

# 手动：
python src/benchmark.py \
    --url http://localhost:10096/asr \
    --audio data/2speakers_example.wav \
    --num-requests 20 --concurrency 20 --max-seconds 30
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--url` | `http://localhost:10096/asr` | 服务端接口 |
| `--audio` | `data/2speakers_example.wav` | 压测音频 |
| `--num-requests` / `--concurrency` | `20` / `20` | 请求总数 / 并发数 |
| `--max-seconds` | 无 | 截取音频前 N 秒（加快压测） |
| `--spk` | 关 | 压测时开启说话人分离 |

输出指标：总耗时、成功/失败、吞吐(req/s)、延迟（平均/最小/最大/P50）、RTF、
平均转写字数。

> 参考：20 并发 × 30s 音频在 TP=2、`gpu_memory_utilization=0.3` 下约 20s 跑完，
> 100% 成功，RTF ≈ 0.034（30 倍实时），延迟瓶颈主要在排队而非算力。
> 想提升吞吐可调高 `--gpu-memory-utilization`（更大 KV cache，批次更大）。

---

## 常见问题

- **`torch==2.9.0+cu129 is not registered` / 找不到这个版本**：必须带 PyTorch 索引，
  `requirements.txt` 顶部已有 `--extra-index-url https://download.pytorch.org/whl/cu129`。
- **`model 'xxx' is not registered`**：多半是模型没下对。确认 `models/` 下三个目录
  都已下载，且说话人模型用的是 `speech_campplus_sv_zh_en_16k-common_advanced`
  （`speech_campplus_speaker-diarization_common` 组合模型不带权重，无法用）。
- **`World size (2) is larger than the available GPUs (1)`**：`CUDA_VISIBLE_DEVICES`
  暴露的卡数 < `--tensor-parallel-size`。用多卡 TP 时把多张卡都放进
  `CUDA_VISIBLE_DEVICES`。
- **显卡被占满 / 并发压测排队**：`serve_vllm.sh`、`pipline_vad_asr.sh` 里的 GPU
  号按实际空闲情况改。
- **`dtype` 建议 `bf16`**：`fp16` 在音频 embedding 路径上可能数值溢出导致乱码；
  无 bf16 支持的卡（如 V100）用 `fp32`。
- **长音频务必先 VAD 切段**：Nano 在过长片段上会退化并出现重复性幻觉，
  分段 + 外扩 + 超长二次切分是离线模式的核心思路。

---

## 参考仓库

- [FunASR (ModelScope)](https://github.com/modelscope/FunASR) —— 本项目的基础框架。
  FSMN-VAD、CAM++ 说话人分离、websocket 流式服务、vLLM 推理实现等均来自或参考
  该仓库（`examples/industrial_data_pretraining/fun_asr_nano/` 下的示例）。
