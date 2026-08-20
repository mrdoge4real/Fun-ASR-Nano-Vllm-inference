export CUDA_VISIBLE_DEVICES=0
python src/pipeline_vad_asr.py \
    --audio data/2speakers_example.wav \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.3 \
    --spk-model models/speech_campplus_sv_zh_en_16k-common_advanced \
    --device cuda:0