#!/usr/bin/env python3
"""Build the GPT2x speech-text pretraining mixture for audio/train_audio.py.

Pulls a ~1B-token slice of Marin's main speech mix (YODAS2-en / Emilia) from HuggingFace,
tokenizes each document's text spans with GPT-2 BPE and audio spans with the Mimi codec ids into
one combined "GPT2x" vocabulary, adds 5% local FineWeb text, shuffles, and writes uint32 shards to
data/audio/marin_mix_gpt2x/ ({prefix}_train.bin / _val.bin / meta.json).

Combined vocab (GPT2x layout):
    [ 0 .. 50256 ]     GPT-2 BPE text tokens
    [ 50257 .. 50262 ] 6 specials (bot, eot, text_start, text_end, audio_start, audio_end)
    [ 50263 .. 66646 ] 16384 Mimi audio tokens

Run from the repo root (needs HF access + local FineWeb shards in data/text/fineweb10B/):
    .venv/bin/python data/audio/marin_audio.py --target_tokens 1e9
"""
import argparse
import json
import os
import re

import numpy as np

MAGIC = 20240930  # shard magic (nanoGPT fineweb uses 20240520)
HEADER_INTS = 256

MIMI_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"
MIMI_AUDIO_BASE = 128260   # first audio-token id in the Mimi tokenizer
MIMI_AUDIO_COUNT = 16384   # 8 codebooks x 2048

GPT2_VOCAB = 50257
SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|text_start|>",
            "<|text_end|>", "<|audio_start|>", "<|audio_end|>"]
SPECIAL_BASE = GPT2_VOCAB
AUDIO_BASE = GPT2_VOCAB + len(SPECIALS)
VOCAB = AUDIO_BASE + MIMI_AUDIO_COUNT
SPECIAL_ID = {s: SPECIAL_BASE + i for i, s in enumerate(SPECIALS)}
BOS, EOS = SPECIAL_ID["<|begin_of_text|>"], SPECIAL_ID["<|end_of_text|>"]
MARKER_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")

# Speech-text pretraining corpora (gated soda-research/* mirrors; need HF access),
# weighted to match Marin Audio's MAIN pretraining mix (exp1699_marin_audio_all):
# Yodas2-En 131 : Emilia-YODAS-En 73 : Emilia-En 37 at 95%, + 5% text-only.
MIX_SOURCES = [
    # (name, hf_repo, path_prefix, weight)
    ("yodas2-en",       "potsawee/yodas2-mm-pretrain", "en",              131),
    ("emilia-yodas-en", "potsawee/emilia-mm-pretrain", "Emilia-YODAS/EN",  73),
    ("emilia-en",       "potsawee/emilia-mm-pretrain", "Emilia/EN",        37),
]
TEXT_FRAC = 0.05  # 5% text-only, matching Marin's Nemotron ratio (FineWeb stand-in)
TEXT_GLOB = "data/text/fineweb10B/fineweb_train_*.bin"  # local GPT-2 uint16 shards


def _write_bin(path, arr, bits=32):
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2], header[3] = MAGIC, 1, len(arr), bits
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())
    print(f"  wrote {len(arr):,} tokens -> {path}")


def _build_audio_char_map():
    """Map each Mimi audio-token's unicode char -> its 0-based audio index."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(hf_hub_download(MIMI_TOKENIZER, "tokenizer.json"))
    id2tok = {v: k for k, v in tk.get_vocab().items()}
    char2idx = {}
    for i in range(MIMI_AUDIO_BASE, MIMI_AUDIO_BASE + MIMI_AUDIO_COUNT):
        tok = id2tok.get(i)
        assert tok is not None and len(tok) == 1, f"audio id {i} -> {tok!r} not a single char"
        char2idx[tok] = i - MIMI_AUDIO_BASE
    for s in SPECIALS:
        assert tk.token_to_id(s) is not None, f"missing special {s} in Mimi tokenizer"
    return char2idx


def _encode_doc(s, gpt2, char2idx):
    """Tokenize one speech-text document into combined-vocab ids: text spans ->
    GPT-2 BPE, audio spans (between <|audio_start|>/<|audio_end|>) -> Mimi audio
    ids, markers -> specials. Documents already begin with <|begin_of_text|>."""
    ids, in_audio = [], False
    for part in MARKER_RE.split(s):
        if not part:
            continue
        if part in SPECIAL_ID:
            ids.append(SPECIAL_ID[part])
            in_audio = part == "<|audio_start|>" or (in_audio and part != "<|audio_end|>")
        elif in_audio:
            for ch in part:
                a = char2idx.get(ch)
                assert a is not None, f"unknown audio char {ch!r}"
                ids.append(AUDIO_BASE + a)
        else:
            ids.extend(gpt2.encode_ordinary(part))
    return ids


def _load_fineweb_docs(glob_pat, n_tokens, doc_len=2048):
    """Read ~n_tokens GPT-2 tokens from local FineWeb .bin shards (256-int32 header,
    uint16 tokens) as ~doc_len-token docs. Ids are 0..50256 -> valid text ids."""
    import glob as _glob
    docs, got = [], 0
    for f in sorted(_glob.glob(glob_pat)):
        arr = np.memmap(f, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4)
        take = int(min(len(arr), n_tokens - got))
        chunk = np.asarray(arr[:take], dtype=np.uint32)
        for i in range(0, len(chunk), doc_len):
            docs.append(chunk[i : i + doc_len])
        got += take
        if got >= n_tokens:
            break
    return docs, got


def prepare_data():
    """Build a ~target-token GPT2x mixture matching Marin's main pretraining mix."""
    import random
    import tiktoken
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=float, default=1e9)
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "marin_mix_gpt2x")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.basename(out_dir.rstrip("/"))

    print("building audio char map from Mimi tokenizer ...")
    char2idx = _build_audio_char_map()
    gpt2 = tiktoken.get_encoding("gpt2")
    print(f"combined vocab {VOCAB}; target {args.target_tokens/1e9:.2f}B tokens")

    speech_total = args.target_tokens * (1 - TEXT_FRAC)
    wsum = sum(w for *_, w in MIX_SOURCES)
    docs, counts = [], {}
    for name, repo, path_prefix, w in MIX_SOURCES:
        budget = speech_total * w / wsum
        files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                       if f.startswith(path_prefix) and f.endswith(".parquet"))
        print(f"[{name}] target {budget/1e6:.0f}M tokens from {repo}:{path_prefix} ({len(files)} files)")
        got = 0
        for f in files:
            if got >= budget:
                break
            p = hf_hub_download(repo, f, repo_type="dataset")
            for txt in pq.read_table(p, columns=["text"]).column("text").to_pylist():
                if not txt:
                    continue
                docs.append(np.asarray(_encode_doc(txt, gpt2, char2idx), dtype=np.uint32))
                got += len(docs[-1])
                if got >= budget:
                    break
            os.remove(p)  # free disk between shards
            print(f"  [{name}] {got/1e6:.0f}M / {budget/1e6:.0f}M")
        counts[name] = int(got)

    # 5% text-only (local FineWeb as a Nemotron stand-in; GPT-2 ids drop straight in)
    tdocs, tgot = _load_fineweb_docs(TEXT_GLOB, args.target_tokens * TEXT_FRAC)
    docs += tdocs
    counts["text-fineweb"] = int(tgot)
    print(f"[text-fineweb] {tgot/1e6:.0f}M tokens")

    # shuffle documents -> mixed stream; hold out val_frac for validation
    random.Random(args.seed).shuffle(docs)
    total = sum(len(d) for d in docs)
    val_target = total * args.val_frac
    val_docs, train_docs, nv = [], [], 0
    for d in docs:
        (val_docs if nv < val_target else train_docs).append(d)
        if nv < val_target:
            nv += len(d)
    for split, dd in [("train", train_docs), ("val", val_docs)]:
        toks = np.concatenate(dd) if dd else np.zeros(0, np.uint32)
        assert not len(toks) or int(toks.max()) < VOCAB
        _write_bin(os.path.join(out_dir, f"{prefix}_{split}.bin"), toks)

    meta = {"mix": "marin main-pretraining ratios (yodas2 131 : emilia-yodas 73 : emilia 37 @95%, +5% text)",
            "sources": counts, "tokenizer": "gpt2x (gpt2 text + mimi audio)", "vocab_size": VOCAB,
            "text_vocab": GPT2_VOCAB, "audio_base": AUDIO_BASE, "bos_id": BOS, "eos_id": EOS,
            "train_tokens": int(total - nv), "val_tokens": int(nv)}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"done -> {out_dir}  sources={counts}  (train {(total-nv)/1e9:.2f}B, val {nv/1e6:.1f}M)")


if __name__ == "__main__":
    prepare_data()
