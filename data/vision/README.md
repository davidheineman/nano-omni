## pull core Molmo2 data

there's two setups:

- `data/vision/molmo2_sft_simple.py` - standalone data ingestion (only a few subsets)
- `data/vision/molmo2_sft_full.py` - uses `molmo2` dataloaders (better coverage; more deps)

```bash
export MOLMO_DATA_DIR=/storage/home/dhei/molmo_data
export HF_HOME=/storage/home/dhei/.cache/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0

python data/vision/molmo2_sft_full.py \
    --out data/vision/molmo2_sft_full \
    --max_train 6000 \
    --max_val 500 \
    --resume \
    --skip "pixmo_ask_model_anything pixmo_cap pixmo_cap_qa_as_user_qa pixmo_multi_image_qa_multi_only_max5 pixmo_multi_points pixmo_points_train pixmo_count_train pixmo_points_high_freq_train"
```

## pull additional data

only partial data is available on HF, some must be pulled manually (to `data/vision/molmo2_manual_pull`):

```sh
pip install gdown "huggingface_hub[cli]"
```

```sh
mkdir -p dvqa plotqa figureqa ai2d

# dv_qa
gdown "https://drive.google.com/uc?id=1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ" -O molmo2_manual_pull/dvqa/dvqa_images.download
gdown "https://drive.google.com/uc?id=1VKYd3kaiCFziSsSv4SgQJ2T5m7jxuh5u" -O molmo2_manual_pull/dvqa/dvqa_annotations.download

# plot_qa
gdown "https://drive.google.com/uc?id=1AYuaPX-Lx7T0GZvnsPgN11Twq2FZbWXL" -O molmo2_manual_pull/plotqa/train_images.download
gdown "https://drive.google.com/uc?id=1i74NRCEb-x44xqzAovuglex5d583qeiF" -O molmo2_manual_pull/plotqa/val_images.download
gdown "https://drive.google.com/uc?id=1D_WPUy91vOrFl6cJUkE55n3ZuB6Qrc4u" -O molmo2_manual_pull/plotqa/test_images.download
gdown "https://drive.google.com/uc?id=1UNvkdq1YJD_ne6D3zbWtoQij37AtfpNp" -O molmo2_manual_pull/plotqa/train_annotations.download
gdown "https://drive.google.com/uc?id=1y9RwXSye2hnX0e2IlfSK34ESbeVblhH_" -O molmo2_manual_pull/plotqa/val_annotations.download
gdown "https://drive.google.com/uc?id=1OQBkoe_dpvFs-jnWAdRdxzh1-hgNd9bO" -O molmo2_manual_pull/plotqa/test_annotations.download

# figure_qa
MS=https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508
for s in train1 validation1 validation2 test1 test2; do
    curl -fL --retry 3 -o molmo2_manual_pull/figureqa/figureqa-$s-v1.tar.gz "$MS/figureqa-$s-v1.tar.gz"
done

# ai2_diagram
curl -fL -o molmo2_manual_pull/ai2d/ai2d-all.zip "http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip"
curl -fL -o molmo2_manual_pull/ai2d/ai2d_test_ids.csv "https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv"
hf download lmms-lab/ai2d --repo-type dataset --local-dir molmo2_manual_pull/ai2d/hf_lmms_lab_ai2d # fallback if S3 404s
```

**gated data:**

here, you must register on an account and manually download:

| dataset | link | dest |
|---|---|
| **DocVQA** (`doc_qa`) | https://rrc.cvc.uab.es/?ch=17 (Task 1) | `molmo2_manual_pull/docqa/` |
| **InfographicVQA** (`info_qa`) | https://rrc.cvc.uab.es/?ch=17 (Task 3) | `molmo2_manual_pull/info_qa/` |
| **ST-VQA** (`st_qa`) | https://rrc.cvc.uab.es/?ch=11 | `molmo2_manual_pull/scene-text/` |

## missing data

| dataset(s) | group | reason |
| entire **demo** group (0.15) | pixmo_ask_model_anything, pixmo_cap, pixmo_cap_qa, pixmo_multi_image_qa, molmo2_human_qa, vixmo captions | PixMo per-example image URLs are ~97% **link-rotted**; molmo2_human_qa / vixmo gated |
| pixmo pointing (in image_pointing) | pixmo_multi_points, pixmo_points_train, pixmo_count_train, pixmo_points_high_freq_train | same PixMo URL rot (cosyn_point covers pointing) |
| entire **video_academic** (0.20), **video_pointing** (0.15), **tracking** (0.15) | — | need TBs of source video + manual clip extraction (Stage 4) |
