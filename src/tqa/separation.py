"""T2 separation analysis v2 (paired design).

Constructs query PAIRS (q1, q2) = 2i queries sharing anchor e and branch r_b:
  q1 = (e, r_a) ∩ (e, r_b)      q2 = (e, r_c) ∩ (e, r_b)
where the model's projection scores for r_a and r_c are nearly identical
(cosine > 0.98 = the executor cannot distinguish the branches) while the true
successor sets differ. Fixed product ops must then emit (nearly) the same
output ranking for q1 and q2 regardless of their different ground truths.
Witness-aware ops (gated t-norm) receive different relation features and may
separate the two.

Metrics per model:
  - Spearman(s_out(q1), s_out(q2)): output collision (fixed -> ~1)
  - MRR on collision-prone pairs vs low-cosine CONTROL pairs
Prediction (T2): fixed has higher output-Spearman; gated MRR gain over fixed
concentrates in the collision-prone group.
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
from tqa.model import TQA

R = "/mnt/ssd/tqa/runs"


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return (ra * rb).mean().item()


def q_mrr(scores, hard, filt, num_ent):
    mask = torch.ones(num_ent, dtype=torch.bool)
    for x in filt:
        mask[x] = False
    sm = scores.clone()
    sm[~mask] = -1e9
    rrs = []
    for e in sorted(hard):
        t = scores[e]
        rrs.append(1.0 / ((sm > t).sum().item() + 1))
    return float(np.mean(rrs)) if rrs else 0.0


@torch.no_grad()
def build_pairs(graph, model, device, n_anchors=150, cos_hi=0.98, rng_seed=0):
    """Find (e, r_a, r_c, r_b) collision pairs + control pairs (low cosine)."""
    rng = random.Random(rng_seed)
    adj = graph.adj_train
    rels = [r for r in range(2 * graph.num_rel) if len(adj[r]) >= 5]
    anchors = rng.sample([e for e in range(graph.num_ent)
                          if sum(1 for r in rels if e in adj[r]) >= 6],
                         min(n_anchors, graph.num_ent))
    coll, ctrl = [], []
    for e in anchors:
        my_rels = [r for r in rels if e in adj[r] and adj[r][e]]
        if len(my_rels) < 4:
            continue
        rng.shuffle(my_rels)
        # batch-compute branch score vectors for this anchor's relations
        rels_cand = my_rels[:10]
        progs = [[("A", 0, e), ("P", 1, 0, r)] for r in rels_cand]
        s = model(graph, progs, device).cpu()  # (k, E)
        succs = [adj[r][e] for r in rels_cand]
        k = len(rels_cand)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = s[i], s[j]
                cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                same_set = succs[i] == succs[j]
                if same_set:
                    continue
                # pick r_b distinct from both, with non-empty common successors
                for m_idx in range(k):
                    if m_idx in (i, j):
                        continue
                    rb = rels_cand[m_idx]
                    if succs[i] & succs[m_idx] and succs[j] & succs[m_idx]:
                        rec = (e, rels_cand[i], rels_cand[j], rb)
                        if cos >= cos_hi:
                            coll.append(rec)
                        elif cos < 0.5:
                            ctrl.append(rec)
                        break
            if len(coll) >= 120 and len(ctrl) >= 120:
                break
        if len(coll) >= 120 and len(ctrl) >= 120:
            break
    return coll[:120], ctrl[:120]


@torch.no_grad()
def eval_pairs(graph, model, pairs, device):
    """Returns list of (spearman, mrr_q1, mrr_q2) per pair."""
    out = []
    num_ent = graph.num_ent
    for (e, ra, rc, rb) in pairs:
        p1 = [("A", 0, e), ("P", 10, 0, ra), ("P", 11, 0, rb), ("I", 1, [10, 11])]
        p2 = [("A", 0, e), ("P", 10, 0, rc), ("P", 11, 0, rb), ("I", 1, [10, 11])]
        s = model(graph, [p1, p2], device).cpu()
        # ground truth on train graph (hard protocol needs full; use train-GT +
        # full-GT difference like standard: recompute answers on full adj)
        a1 = graph.adj_train[ra].get(e, set()) & graph.adj_train[rb].get(e, set())
        a2 = graph.adj_train[rc].get(e, set()) & graph.adj_train[rb].get(e, set())
        f1 = graph.adj_full[ra].get(e, set()) & graph.adj_full[rb].get(e, set())
        f2 = graph.adj_full[rc].get(e, set()) & graph.adj_full[rb].get(e, set())
        hard1, hard2 = f1 - a1, f2 - a2
        if not hard1 or not hard2:
            # fall back to train answers (both models judged identically)
            hard1, hard2 = a1 - {e}, a2 - {e}
            f1, f2 = a1 | {e}, a2 | {e}
        if not hard1 or not hard2:
            continue
        sp = spearman(s[0], s[1])
        m1 = q_mrr(s[0], hard1, f1 | {e}, num_ent)
        m2 = q_mrr(s[1], hard2, f2 | {e}, num_ent)
        out.append((sp, m1, m2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name:ops pairs, e.g. p2_fixed_s42_50k:fixed p2_gated_s42_50k:gated")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    graphs = load_graphs(("fb15k237",))
    g = graphs["fb15k237"]

    # pairs must be built ONCE with a reference model (the fixed executor's own
    # projections define "indistinguishable branches"); reused for all models
    first_name, first_ops = args.ckpts[0].split(":")
    ref = TQA(d=64, ops=first_ops).to(device)
    ref.load_state_dict(torch.load(os.path.join(R, f"{first_name}.pt"), map_location=device))
    ref.eval()
    coll, ctrl = build_pairs(g, ref, device)
    print(f"pairs: collision={len(coll)} control={len(ctrl)} (reference={first_name})",
          flush=True)

    report = {"collision_pairs": len(coll), "control_pairs": len(ctrl), "models": {}}
    for spec in args.ckpts:
        name, ops = spec.split(":")
        m = TQA(d=64, ops=ops).to(device)
        m.load_state_dict(torch.load(os.path.join(R, f"{name}.pt"), map_location=device))
        m.eval()
        rows_c = eval_pairs(g, m, coll, device)
        rows_k = eval_pairs(g, m, ctrl, device)
        def agg(rows):
            if not rows:
                return None
            sp = np.mean([r[0] for r in rows])
            mr = np.mean([np.mean([r[1], r[2]]) for r in rows])
            return {"spearman": round(float(sp), 4), "MRR": round(float(mr), 4),
                    "n": len(rows)}
        rec = {"collision": agg(rows_c), "control": agg(rows_k)}
        report["models"][name] = rec
        print(f"{name}: collision {rec['collision']} | control {rec['control']}",
              flush=True)

    with open(os.path.join(R, "separation_v2.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("saved runs/separation_v2.json", flush=True)


if __name__ == "__main__":
    main()
