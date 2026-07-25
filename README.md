a derivative of https://github.com/KellerJordan/modded-nanogpt for text-vision-speech models.

### branches

- [github.com/davidheineman/nano-omni/tree/finetune](https://github.com/davidheineman/nano-omni/tree/finetune) - pretrain + fine-tune existing record runs

### setup

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt

uv venv --python 3.12 .venv
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

### flash attenion 3 install (optional)

```bash
# only for 2025 `*_FA3` runs
uv venv --python 3.12 .venv_fa3
uv sync --extra all
export VIRTUAL_ENV=$PWD/.venv_fa3
uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps wheels/flash_attn_3-built-torch290cu128.whl
.venv_fa3/bin/python -c "from flash_attn_interface import flash_attn_varlen_func; print('FA3 OK')"
```
