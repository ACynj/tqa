"""Phase-3 standalone evaluation:
1) negation query types (zero-shot on p2 checkpoints + LOO checkpoints)
2) collision-bucket separation analysis (T2 empirical test)
Usage: python -m tqa.eval_extra --ckpt runs/p2_gated_s42_50k.pt --ops gated ...
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqa.kgdata import load_graphs
from tqa.querygen import NEG_TYPES, EPFO_TYPES, gen_pool
from tqa.model import TQA
from tqa.train import get_eval_pool, evaluate

CACHE = "/mnt/ssd/tqa/data/queries"


def get_neg_pool(graph, qtype, n, seed):
    f = os.path.join(CACHE, f"{graph.name}_{qtype}_eval_{n}_s{seed}.json")
    if os.path.exists(f):
        raw = json.load(open(f))
        return [{"prog": [tuple(x) for x in it["prog"]], "anchors": it["anchors"],
                 "easy": set(it["easy"]), "hard": set(it["hard"]),
                 "filter": set(it["filter"]), "type": qtype} for it in raw]
    rng = random.Random(seed)
    pool = gen_pool(graph, qtype, n, rng, hard=True)
    ser = [{"prog": [list(op) for op in it["prog"]], "anchors": it["anchors"],
            "easy": sorted(it["easy"]), "hard": sorted(it["hard"]),
            "filter": sorted(it["filter"])} for it in pool]
    with open(f, "w") as fh:
        json.dump(ser, fh)
    return pool


@torch.no_grad()
def collision_analysis(model, graph, pools, device, log=print):
    """T2 empirical test: bucket 2i queries by branch-score cosine similarity
    (computed with the model's own projections). Fixed product ops must give
    identical outputs when cos->1; witness-aware ops can differ. Report the
    SQA gain per bucket using cached per-query MRR... simplified: report the
    score-difference between paired branches and per-bucket MRR."""
    from torch.nn.functional import cosine_similarity
    model.eval()
    qt = "2i"
    pool = pools[qt][:300]
    recs = []
    for i in range(0, len(pool), 16):
        chunk = pool[i:i + 16]
        # run first branch only: (A, P) -> get branch scores per query
        # we approximate branch marginals via two 1p-style runs
        progs1, progs2 = [], []
        for it in chunk:
            prog = it["prog"]
            # 2i prog: [A0, P r_a 0->10, P r_b 0->11, I [10,11]->1]
            e = prog[0][2]
            r_a, r_b = prog[1][3], prog[2][3]
            progs1.append([("A", 0, e), ("P", 1, 0, r_a)])
            progs2.append([("A", 0, e), ("P", 1, 0, r_b)])
        s_a = model(graph, progs1, device).cpu()
        s_b = model(graph, progs2, device).cpu()
        full = model(graph, [it["prog"] for it in chunk], device).cpu()
        for j, it in enumerate(chunk):
            cos = cosine_similarity(s_a[j], s_b[j], dim=0).item()
            # per-hard-answer MRR for this query
            filt = set(it["filter"]) | set(it["anchors"])
            mask = torch.ones(graph.num_ent, dtype=torch.bool)
            for e in filt:
                mask[e] = False
            sm = full[j].clone()
            sm[~mask] = -1e9
            rrs = []
            for e in sorted(it["hard"]):
                t = full[j, e]
                rrs.append(1.0 / ((sm > t).sum().item() + 1))
            recs.append((cos, float(np.mean(rrs)) if rrs else 0.0))
    recs = np.array(recs)
    order = np.argsort(recs[:, 0])
    recs = recs[order]
    q = np.quantile(recs[:, 0], [0.25, 0.5, 0.75])
    buckets = [recs[recs[:, 0] <= q[0]], recs[(recs[:, 0] > q[0]) & (recs[:, 0] <= q[1])],
               recs[(recs[:, 0] > q[1]) & (recs[:, 0] <= q[2])], recs[recs[:, 0] > q[2]]]
    out = []
    for bi, b in enumerate(buckets):
        c_lo, c_hi = b[:, 0].min(), b[:, 0].max()
        mrr = b[:, 1].mean()
        out.append({"bucket": bi, "cos_lo": round(float(c_lo), 3), "cos_hi": round(float(c_hi), 3),
                    "MRR": round(float(mrr), 4), "n": int(len(b))})
        log(f"    bucket{bi} cos[{c_lo:.3f},{c_hi:.3f}] MRR={mrr:.4f} n={len(b)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ops", choices=["fixed", "learned", "gated"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--graph", default="fb15k237")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--neg-only", action="store_true")
    ap.add_argument("--no-neg", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    graphs = load_graphs((args.graph,))
    g = graphs[args.graph]

    model = TQA(d=args.d, ops=args.ops).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    tag = args.tag or os.path.basename(args.ckpt).replace(".pt", "")

    result = {"ckpt": args.ckpt, "ops": args.ops, "graph": args.graph}

    if not args.no_neg:
        neg_pools = {}
        print(f"[neg eval] {tag}", flush=True)
        for qt in NEG_TYPES:
            neg_pools[qt] = get_neg_pool(g, qt, 300, 778)
        per_type, avg = evaluate(model, g, neg_pools, device)
        result["negation"] = {"per_type": per_type, "avg": avg}

    if not args.neg_only:
        epfo_pools = {qt: get_eval_pool(g, qt, 500, 777) for qt in ("2i",)}
        print(f"[collision analysis] {tag}", flush=True)
        result["collision_2i"] = collision_analysis(model, g, epfo_pools, device)

    outdir = "/mnt/ssd/tqa/runs"
    with open(os.path.join(outdir, f"extra_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"saved runs/extra_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
