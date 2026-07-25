a derivative of https://github.com/KellerJordan/modded-nanogpt for text-vision-speech models.

### branches

- [github.com/davidheineman/nano-omni/tree/finetune](https://github.com/davidheineman/nano-omni/tree/finetune) - pretrain + fine-tune existing record runs

### setup

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt
uv sync --extra all

# pull data
python data/cached_fineweb10B.py 9 # first 900M tokens
python data/vision/molmo2_sft.py
python data/audio/marin_audio.py

# train text model
torchrun --standalone --nproc_per_node=8 train_gpt.py

# train text-vision model
python convert_siglip.py \
    --model google/siglip-so400m-patch14-384 --out siglip_so400m_378.pt
torchrun --standalone --nproc_per_node=1 vision/train_vision.py

# train text-audio model
torchrun --standalone --nproc_per_node=8 audio/train_audio.py

# (see SETUP.md for FlashAttention details / running all record runs)
```
