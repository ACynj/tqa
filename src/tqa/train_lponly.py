"""TQA training / evaluation driver.

Usage:
  python -m tqa.train --ops fixed   --gpu 0   # SQA-1
  python -m tqa.train --ops learned --gpu 1   # SQA-2
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqa.kgdata import load_graphs
from tqa.querygen import gen_pool
EPFO_TYPES = ["1p"]
from tqa.model import TQA

CACHE = "/mnt/ssd/tqa/data/queries"


def get_train_pool(graph, qtype, n, rng_seed, cache=True):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{graph.name}_{qtype}_{n}_s{rng_seed}.json")
    if cache and os.path.exists(f):
        with open(f) as fh:
            raw = json.load(fh)
        return [{"prog": [tuple(x) for x in it["prog"]],
                 "anchors": it["anchors"], "answers": set(it["answers"]),
                 "type": qtype} for it in raw]
    rng = random.Random(rng_seed)
    pool = gen_pool(graph, qtype, n, rng, hard=False)
    if cache:
        ser = [{"prog": [list(op) for op in it["prog"]], "anchors": it["anchors"],
                "answers": sorted(it["answers"])} for it in pool]
        with open(f, "w") as fh:
            json.dump(ser, fh)
    return pool


def get_eval_pool(graph, qtype, n, rng_seed):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{graph.name}_{qtype}_eval_{n}_s{rng_seed}.json")
    if os.path.exists(f):
        with open(f) as fh:
            raw = json.load(fh)
        return [{"prog": [tuple(x) for x in it["prog"]], "anchors": it["anchors"],
                 "easy": set(it["easy"]), "hard": set(it["hard"]),
                 "filter": set(it["filter"]), "type": qtype} for it in raw]
    rng = random.Random(rng_seed)
    pool = gen_pool(graph, qtype, n, rng, hard=True)
    ser = [{"prog": [list(op) for op in it["prog"]], "anchors": it["anchors"],
            "easy": sorted(it["easy"]), "hard": sorted(it["hard"]),
            "filter": sorted(it["filter"])} for it in pool]
    with open(f, "w") as fh:
        json.dump(ser, fh)
    return pool


def batch_loss(scores, items, device, num_ent, k_neg=256):
    """Ranking CE + BCE over sampled negatives. scores: (B, E)."""
    B, E = scores.shape
    loss = 0.0
    for b, it in enumerate(items):
        pos = torch.tensor(sorted(it["answers"]), dtype=torch.long, device=device)
        neg = torch.randint(0, E, (k_neg,), device=device)
        s_pos = scores[b, pos].mean()
        s_neg = scores[b, neg]
        ce = -(s_pos - torch.logsumexp(torch.cat([s_pos[None], s_neg]), dim=0))
        bce = F_bce(scores[b, pos], scores[b, neg])
        loss = loss + ce + 0.5 * bce
    return loss / B


def F_bce(sp, sn):
    return -(torch.log(sp + 1e-9).mean() + torch.log(1 - sn + 1e-9).mean()) / 2


@torch.no_grad()
def evaluate(model, graph, pools, device, batch=32, max_q=None, log=print):
    model.eval()
    per_type = {}
    for qt, pool in pools.items():
        if max_q:
            pool = pool[:max_q]
        rr_all, h1, h3, h10, cnt = [], 0, 0, 0, 0
        rr_easy, cnt_easy = [], 0
        for i in range(0, len(pool), batch):
            chunk = pool[i:i + batch]
            scores = model(graph, [c["prog"] for c in chunk], device)
            scores = scores.cpu()
            for j, it in enumerate(chunk):
                filt = set(it["filter"]) | set(it["anchors"])
                cand_mask = torch.ones(graph.num_ent, dtype=torch.bool)
                for e in filt:
                    cand_mask[e] = False
                s_masked = scores[j].clone()
                s_masked[~cand_mask] = -1e9
                hard = sorted(it["hard"])
                ranks = []
                for e in hard:
                    target = scores[j, e]  # unmasked value of this answer
                    rank = (s_masked > target).sum().item() + 1
                    ranks.append(rank)
                rr_all.extend(1.0 / r for r in ranks)
                h1 += sum(r <= 1 for r in ranks)
                h3 += sum(r <= 3 for r in ranks)
                h10 += sum(r <= 10 for r in ranks)
                cnt += len(ranks)
                # easy-answer reference metrics (derivable from observed graph)
                for e in sorted(it["easy"]):
                    target = scores[j, e]
                    rank = (s_masked > target).sum().item() + 1
                    rr_easy.append(1.0 / rank)
                cnt_easy += len(it["easy"])
        mrr = float(np.mean(rr_all)) if rr_all else 0.0
        per_type[qt] = {
            "MRR": round(mrr, 4),
            "H@1": round(h1 / max(cnt, 1), 4),
            "H@3": round(h3 / max(cnt, 1), 4),
            "H@10": round(h10 / max(cnt, 1), 4),
            "MRR_easy": round(float(np.mean(rr_easy)), 4) if rr_easy else 0.0,
            "n_q": len(pool), "n_ans": cnt, "n_easy": cnt_easy,
        }
        log(f"    [{qt}] MRR={per_type[qt]['MRR']:.4f} H@1={per_type[qt]['H@1']:.4f} "
            f"H@3={per_type[qt]['H@3']:.4f} H@10={per_type[qt]['H@10']:.4f} "
            f"MRR_easy={per_type[qt]['MRR_easy']:.4f} "
            f"(queries={len(pool)}, answers={cnt})")
    avg = {m: round(float(np.mean([v[m] for v in per_type.values()])), 4)
           for m in ("MRR", "H@1", "H@3", "H@10")}
    model.train()
    return per_type, avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", choices=["fixed", "learned", "gated", "gated2"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-per-type", type=int, default=4000)
    ap.add_argument("--eval-per-type", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    run_name = args.run_name or f"sqa1_{args.ops}_seed{args.seed}"
    logdir = "/mnt/ssd/tqa/logs"
    os.makedirs(logdir, exist_ok=True)
    logf = open(os.path.join(logdir, f"{run_name}.log"), "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True

    log(f"===== TQA run: ops={args.ops} steps={args.steps} d={args.d} "
        f"batch={args.batch} seed={args.seed} =====")

    log("loading graphs (fb15k237, wn18rr, codex-m) ...")
    graphs = load_graphs()
    for n, g in graphs.items():
        log(f"  {n}: |E|={g.num_ent} |R|={g.num_rel} train={len(g.split_triples['train'])}")

    log("generating / loading query pools ...")
    t0 = time.time()
    train_pools = {}
    for n, g in graphs.items():
        for qt in EPFO_TYPES:
            train_pools[(n, qt)] = get_train_pool(g, qt, args.train_per_type,
                                                  args.seed + 137 * (EPFO_TYPES.index(qt) + 1))
    eval_graph = graphs["fb15k237"]
    eval_pools = {qt: get_eval_pool(eval_graph, qt, args.eval_per_type, 777)
                  for qt in EPFO_TYPES}
    log(f"pools ready in {time.time()-t0:.0f}s; eval sizes: "
        + ", ".join(f"{qt}:{len(v)}" for qt, v in eval_pools.items()))

    model = TQA(d=args.d, ops=args.ops).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"model params: {nparam:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    keys = list(train_pools.keys())
    rng = random.Random(args.seed)
    model.train()
    step, t_start = 0, time.time()
    losses = []
    while step < args.steps:
        n, qt = rng.choice(keys)
        pool = train_pools[(n, qt)]
        g = graphs[n]
        items = rng.sample(pool, min(args.batch, len(pool)))
        scores = model(g, [it["prog"] for it in items], device)
        loss = batch_loss(scores, items, device, g.num_ent)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            log(f"step {step}/{args.steps} loss={np.mean(losses[-100:]):.4f} "
                f"({step/(time.time()-t_start):.1f} it/s)")
        if step % args.eval_every == 0:
            log(f"-- eval @ step {step} (subset) --")
            evaluate(model, eval_graph, eval_pools, device, max_q=100, log=log)

    log("===== FINAL EVAL (full) =====")
    per_type, avg = evaluate(model, eval_graph, eval_pools, device, log=log)
    log(f"AVG: MRR={avg['MRR']} H@1={avg['H@1']} H@3={avg['H@3']} H@10={avg['H@10']}")
    out = {"args": vars(args), "per_type": per_type, "avg": avg}
    resdir = "/mnt/ssd/tqa/runs"
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, f"{run_name}_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    torch.save(model.state_dict(), os.path.join(resdir, f"{run_name}.pt"))
    log(f"saved results to runs/{run_name}_results.json")
    logf.close()


if __name__ == "__main__":
    main()
