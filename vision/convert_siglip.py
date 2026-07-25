"""Standalone SigLIP -> SiglipViT weight converter (no olmo dependency).

Loads a HuggingFace SigLIP vision tower and remaps it to the parameter names used by
train_vision.py::SiglipViT, interpolating position embeddings to the 27x27 (729-token) grid.

    python convert_siglip.py --model google/siglip-so400m-patch14-384 --out siglip_so400m_378.pt
"""
import argparse
import math

import torch
import torch.nn.functional as F


def convert(model_id: str, out_path: str, target_side: int = 27):
    from transformers import SiglipVisionModel
    m = SiglipVisionModel.from_pretrained(model_id)
    sd = m.state_dict()
    nlayers = m.config.num_hidden_layers
    # transformers>=5 SiglipVisionModel.state_dict() has no "vision_model." prefix
    P = "vision_model." if any(k.startswith("vision_model.") for k in sd) else ""

    out = {}
    # patch embedding: Conv2d [D,3,14,14] -> Linear weight [D, 14*14*3] (h,w,c flatten to match preprocessor)
    pe = sd[P + "embeddings.patch_embedding.weight"]
    out["patch_embedding.weight"] = pe.permute(0, 2, 3, 1).reshape(pe.shape[0], -1).contiguous()
    out["patch_embedding.bias"] = sd[P + "embeddings.patch_embedding.bias"]

    # position embedding: [Pos, D] -> interpolate to target_side^2
    pos = sd[P + "embeddings.position_embedding.weight"]
    npos, D = pos.shape
    side = int(round(math.sqrt(npos)))
    target = target_side * target_side
    if npos != target:
        g = pos.reshape(1, side, side, D).permute(0, 3, 1, 2).float()
        g = F.interpolate(g, size=(target_side, target_side), mode="bicubic",
                          align_corners=False, antialias=True)
        pos = g.permute(0, 2, 3, 1).reshape(target, D).to(pos.dtype)
        print(f"interpolated pos emb {npos} ({side}x{side}) -> {target} ({target_side}x{target_side})")
    out["positional_embedding"] = pos.contiguous()

    for i in range(nlayers):
        src = f"{P}encoder.layers.{i}."
        dst = f"transformer.resblocks.{i}."
        for a, b in [("self_attn.q_proj", "attention.wq"), ("self_attn.k_proj", "attention.wk"),
                     ("self_attn.v_proj", "attention.wv"), ("self_attn.out_proj", "attention.wo"),
                     ("mlp.fc1", "feed_forward.w1"), ("mlp.fc2", "feed_forward.w2"),
                     ("layer_norm1", "attention_norm"), ("layer_norm2", "ffn_norm")]:
            out[dst + b + ".weight"] = sd[src + a + ".weight"]
            out[dst + b + ".bias"] = sd[src + a + ".bias"]

    torch.save(out, out_path)
    print(f"saved {len(out)} tensors -> {out_path} (nlayers={nlayers}, dim={D})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--out", default="results/vision/siglip_so400m_378.pt")
    a = ap.parse_args()
    convert(a.model, a.out)
