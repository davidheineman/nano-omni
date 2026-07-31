import argparse
import glob
import json
import math
import os

# GPT-2 BPE files, read offline on compute nodes from a repo-local cache (populated once,
# on shared /storage). Derived from __file__ so it's not a hardcoded absolute path.
os.environ.setdefault("TIKTOKEN_CACHE_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tiktoken_cache"))

import torch
import torch.nn.functional as F

TBD = "/datasets/pretraining_data/dhei/dolma3_tbd_format/text-ppl"


def discover_sets(tbd=TBD):
    """Map every text-ppl subset -> its jsonl, keyed by HF subset name."""
    sets = {}
    for f in sorted(glob.glob(os.path.join(tbd, "*.jsonl"))):
        base = os.path.basename(f)[:-len(".jsonl")]
        name = base[:-len("_span_ppl")] if base.endswith("_span_ppl") else base
        sets[name] = f
    return sets


EVAL_SETS = discover_sets()
DEFAULT_SETS = [
    "minerva",
    "mbpp",
    "mmlu",
    "dolmino_pool_stack_edu_fim_Python",
    "paloma_m2d2_wikipedia_unsplit"
]

_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("gpt2")
    return _ENC


def _load(path, max_docs):
    docs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        text, spans = d.get("text", ""), d.get("spans") or []
        if not text:
            continue
        docs.append((text, int(spans[0][0]) if spans else 0))  # c0=0 => plain LM ppl
        if max_docs and len(docs) >= max_docs:
            break
    return docs


def _prep(text, c0, max_len):
    e = _enc()
    full = e.encode_ordinary(text)
    cont_start = max(0, min(len(e.encode_ordinary(text[:c0])), len(full) - 1))
    if len(full) > max_len:
        drop = len(full) - max_len
        full, cont_start = full[drop:], max(cont_start - drop, 0)
    cont_bytes = len(e.decode(full[cont_start:]).encode("utf-8"))
    return full, cont_start, cont_bytes


@torch.no_grad()
def score(model, device, path, max_docs=None, max_len=1024, dtype=torch.bfloat16):
    model.eval()
    tot_nats = tot_tok = tot_bytes = used = 0
    for text, c0 in _load(path, max_docs):
        ids, cont_start, cbytes = _prep(text, c0, max_len)
        if len(ids) < 2 or cont_start >= len(ids):
            continue
        x = torch.tensor(ids[:-1], device=device)[None]
        y = torch.tensor(ids[1:], device=device)[None]
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model.forward_logits(x).float()[0]
        j0 = max(cont_start - 1, 0)
        ce = F.cross_entropy(logits[j0:], y[0, j0:], reduction="sum").item()
        n = y[0, j0:].numel()
        if n <= 0 or not math.isfinite(ce):
            continue
        tot_nats += ce
        tot_tok += n
        tot_bytes += cbytes
        used += 1
    if tot_tok == 0:
        return dict(bpb=float("nan"), ppl=float("nan"), n_docs=0)
    return dict(bpb=(tot_nats / math.log(2)) / max(tot_bytes, 1),
                ppl=math.exp(tot_nats / tot_tok), n_docs=used, n_tokens=int(tot_tok))


def score_sets(model, device, sets=None, max_docs=2000, max_len=1024):
    out = {}
    for name in (sets or DEFAULT_SETS):
        m = score(model, device, EVAL_SETS[name], max_docs, max_len)
        out[f"{name}_bpb"], out[f"{name}_ppl"] = m["bpb"], m["ppl"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--list", action="store_true", help="list all registered subset names")
    args = ap.parse_args()
    if args.list:
        print(f"{len(EVAL_SETS)} subsets registered:")
        print("\n".join(sorted(EVAL_SETS)))
        return
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "text_mixing"))
    import train_swarm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_swarm.load_model(args.ckpt, device)
    print(json.dumps(score_sets(model, device, args.sets, args.max_docs), indent=2))


if __name__ == "__main__":
    main()
