"""Synthetic two-world experiment v2: clean constructive demonstration of T2.

Blocks: anchor a with r_a -> {t1..t4}. Two ambiguous relations r1/r2: in each
block r1 covers a hidden half, r2 the other (which half is random per block).
Global semantics are consistent: r1-tails always carry type marker tau1, r2
tails tau2 (visible via demonstrator blocks). At TEST, the query anchor's own
r1/r2 edges are UNOBSERVED. Query (a, r_a) AND (a, r_q):
  - score-factored fuzzy executor: P_{r_q} from a sees no edges -> flat scores
    -> collision; best it can do is rank the 4 targets by r_a evidence alone.
  - order-2 witness executor: scores targets by GLOBAL relation profile match
    (does the target carry tau_{r_q}?) -> exact answer.
Deterministic rule-based executors, no training confound.
"""
import random

def build_world(seed, n_blocks=200, hide_ratio=0.5):
    rng = random.Random(seed)
    ents = 0
    def new():
        nonlocal ents
        ents += 1
        return ents - 1
    edges = []          # (h, r, t); r: 0=r_a, 1=r1, 2=r2, 3=tau1, 4=tau2
    queries = []        # (anchor, r_q, answer_set, all_targets)
    for b in range(n_blocks):
        a = new()
        ts = [new() for _ in range(4)]
        for t in ts:
            edges.append((a, 0, t))
        half1, half2 = ts[:2], ts[2:]
        if rng.random() < 0.5:
            r1_set, r2_set = half1, half2
        else:
            r1_set, r2_set = half2, half1
        for t in r1_set:
            edges.append((t, 3, new()))   # tau1 marker on r1-type targets
        for t in r2_set:
            edges.append((t, 4, new()))   # tau2 marker
        hide = rng.random() < hide_ratio
        if not hide:                      # demonstrator blocks keep local edges
            for t in r1_set:
                edges.append((a, 1, t))
            for t in r2_set:
                edges.append((a, 2, t))
        else:                             # query blocks: local r1/r2 hidden
            rq = rng.choice([1, 2])
            ans = r1_set if rq == 1 else r2_set
            queries.append((a, rq, ans, ts))
    return edges, queries, ents


def fuzzy_executor(edges, a, rq, ts):
    """Score-factored: propagation along observed rq edges from a (none) ->
    flat branch; ranks by r_a support only -> ties among ts."""
    return {t: 1.0 for t in ts}   # all tied -> random order


def order2_executor(edges, a, rq, ts):
    """Witness/order-2: branch score from GLOBAL relation profile: does the
    target carry the type marker of rq (tau1 for r1, tau2 for r2)?"""
    marker = 3 if rq == 1 else 4
    has = {h for h, r, t in edges if r == marker}
    return {t: (1.0 if t in has else 0.0) for t in ts}


def mrr(ranking, answers):
    for i, t in enumerate(ranking, 1):
        if t in answers:
            return 1.0 / i
    return 0.0


for seed in (1, 2, 3):
    edges, queries, n = build_world(seed)
    rng = random.Random(seed)
    fz, o2 = [], []
    for a, rq, ans, ts in queries:
        ts_ = ts[:]
        rng.shuffle(ts_)
        fz.append(mrr(ts_, ans))                     # fuzzy: tied -> random
        o2_r = order2_executor(edges, a, rq, ts)
        o2.append(mrr(sorted(ts_, key=lambda t: -o2_r[t]), ans))
    print(f"seed{seed}: blocks={len(queries)}  fuzzy(得分算子) MRR={sum(fz)/len(fz):.3f}  "
          f"order-2(见证算子) MRR={sum(o2)/len(o2):.3f}", flush=True)
print("SYNTHETIC V2 DONE: 固定得分算子在隐藏边世界必然碰撞(≈随机)，"
      "阶数-2见证算子完全分离")
