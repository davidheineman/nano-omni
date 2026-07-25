a derivative of https://github.com/KellerJordan/modded-nanogpt for text-vision-speech models.

### branches

- [github.com/davidheineman/nano-omni/tree/finetune](https://github.com/davidheineman/nano-omni/tree/finetune) - pretrain + fine-tune existing record runs

### setup

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
# downloads only the first 900M training tokens to save time
python data/cached_fineweb10B.py 9
torchrun --standalone --nproc_per_node=8 train_gpt.py

# (see SETUP.md for FlashAttention details / running all record runs)
```
