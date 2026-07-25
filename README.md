a derivative of https://github.com/KellerJordan/modded-nanogpt for text-vision-speech models.

### setup

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
# downloads only the first 900M training tokens to save time
python data/cached_fineweb10B.py 9
./run.sh
```
