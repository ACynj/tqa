"""EPFO query generation with symbolic ground-truth execution.

Query types (BetaE convention, negation types deferred to phase 2):
  1p, 2p, 2i, 3i, ip, pi, 2u, up  (+1p as chain of length 1)

Each query is a program: list of ops over set-variables.
  ("A", var, ent)                    anchor
  ("P", out, var, rel)               projection along rel (rel may be inverse)
  ("I", out, [vars])                 intersection
  ("U", out, [vars])                 union
Ground truth runs symbolically over adjacency dicts.
"""
import random
import numpy as np

EPFO_TYPES = ["1p", "2p", "2i", "3i", "ip", "pi", "2u", "up"]
NEG_TYPES = ["2in", "3in", "inp", "pin", "pni"]


def _succ(adj, rel, nodes):
    """successors of a node-set along relation rel."""
    d = adj[rel]
    out = set()
    for n in nodes:
        s = d.get(n)
        if s:
            out.update(s)
    return out


def _run_program(prog, adj, universe=None):
    env = {}
    for op in prog:
        if op[0] == "A":
            env[op[1]] = {op[2]}
        elif op[0] == "P":
            env[op[1]] = _succ(adj, op[3], env[op[2]])
        elif op[0] == "I":
            sets = [env[v] for v in op[2]]
            env[op[1]] = set.intersection(*sets)
        elif op[0] == "U":
            sets = [env[v] for v in op[2]]
            env[op[1]] = set.union(*sets)
        elif op[0] == "N":
            env[op[1]] = (universe if universe is not None else set()) - env[op[2]]
    return env


def _sample_rel_with_succ(adj, rels_with_deg, nodes, rng, exclude=()):
    """sample a relation whose successor set on `nodes` is non-empty."""
    cand = [r for r in rels_with_deg if r not in exclude]
    rng.shuffle(cand)
    for r in cand:
        s = _succ(adj, r, nodes)
        if s:
            return r, s
    return None, None


def gen_query(graph, qtype, rng, max_answers=1000):
    """Sample one query of given type; return (program, anchors) or None on failure."""
    adj = graph.adj_train
    nrel = graph.num_rel
    nent = graph.num_ent
    # relations with decent degree for sampling
    rels_with_deg = [r for r in range(2 * nrel) if len(adj[r]) >= 3]
    if len(rels_with_deg) < 8:
        return None

    def rand_ents_with_rel(rel, k):
        heads = list(adj[rel].keys())
        rng.shuffle(heads)
        picked = []
        for h in heads:
            picked.append(h)
            if len(picked) == k:
                break
        return picked

    anchors = []
    if qtype == "1p" or qtype == "2p":
        r1 = rng.choice(rels_with_deg)
        e = rng.choice(list(adj[r1].keys()))
        prog = [("A", 0, e), ("P", 1, 0, r1)]
        anchors = [e]
        if qtype == "2p":
            mids = _succ(adj, r1, {e})
            if not mids:
                return None
            m = rng.choice(list(mids))
            r2 = None
            for r in rng.sample(rels_with_deg, len(rels_with_deg)):
                s = _succ(adj, r, {m})
                if s:
                    r2 = r
                    break
            if r2 is None:
                return None
            prog.append(("P", 2, 1, r2))
        tgt = prog[-1][1]
    elif qtype in ("2i", "3i"):
        nb = 2 if qtype == "2i" else 3
        e = None
        # anchor with >= nb distinct relations having successors
        for _ in range(50):
            r0 = rng.choice(rels_with_deg)
            cand_e = rng.choice(list(adj[r0].keys()))
            rs = [r for r in rels_with_deg if _succ(adj, r, {cand_e})]
            if len(rs) >= nb:
                e = cand_e
                break
        if e is None:
            return None
        rs = [r for r in rels_with_deg if _succ(adj, r, {e})]
        chosen = rng.sample(rs, nb)
        prog = [("A", 0, e)]
        vs = []
        for i, r in enumerate(chosen):
            prog.append(("P", 10 + i, 0, r))
            vs.append(10 + i)
        prog.append(("I", 1, vs))
        tgt = 1
        anchors = [e]
    elif qtype == "ip":
        e1, e2 = None, None
        for _ in range(50):
            r0 = rng.choice(rels_with_deg)
            a, b = rng.sample(list(adj[r0].keys()), 2) if len(adj[r0]) >= 2 else (None, None)
            if a is None:
                continue
            ra = [r for r in rng.sample(rels_with_deg, min(20, len(rels_with_deg)))
                  if _succ(adj, r, {a})]
            rb = [r for r in rng.sample(rels_with_deg, min(20, len(rels_with_deg)))
                  if _succ(adj, r, {b})]
            if ra and rb:
                e1, e2, r1, r2 = a, b, rng.choice(ra), rng.choice(rb)
                break
        if e1 is None:
            return None
        s1 = _succ(adj, r1, {e1})
        s2 = _succ(adj, r2, {e2})
        common = s1 & s2
        if not common:
            return None
        r3 = None
        for r in rng.sample(rels_with_deg, len(rels_with_deg)):
            if _succ(adj, r, common):
                r3 = r
                break
        if r3 is None:
            return None
        prog = [("A", 0, e1), ("P", 10, 0, r1), ("A", 2, e2), ("P", 11, 2, r2),
                ("I", 3, [10, 11]), ("P", 1, 3, r3)]
        tgt = 1
        anchors = [e1, e2]
    elif qtype == "pi":
        r1 = rng.choice(rels_with_deg)
        e = rng.choice(list(adj[r1].keys()))
        mids = _succ(adj, r1, {e})
        if not mids:
            return None
        m = rng.choice(list(mids))
        rs = [r for r in rng.sample(rels_with_deg, min(30, len(rels_with_deg)))
              if _succ(adj, r, {m})]
        if len(rs) < 2:
            return None
        r2, r3 = rng.sample(rs, 2)
        prog = [("A", 0, e), ("P", 1, 0, r1), ("P", 10, 1, r2), ("P", 11, 1, r3),
                ("I", 2, [10, 11])]
        tgt = 2
        anchors = [e]
    elif qtype in ("2u", "up"):
        e1, e2, r1, r2 = None, None, None, None
        for _ in range(50):
            r1 = rng.choice(rels_with_deg)
            r2 = rng.choice(rels_with_deg)
            e1 = rng.choice(list(adj[r1].keys()))
            e2 = rng.choice(list(adj[r2].keys()))
            if _succ(adj, r1, {e1}) and _succ(adj, r2, {e2}):
                break
        else:
            return None
        prog = [("A", 0, e1), ("P", 10, 0, r1), ("A", 2, e2), ("P", 11, 2, r2),
                ("U", 3, [10, 11])]
        tgt = 3
        anchors = [e1, e2]
        if qtype == "up":
            common = _run_program(prog, adj)[3]
            r3 = None
            for r in rng.sample(rels_with_deg, len(rels_with_deg)):
                if _succ(adj, r, common):
                    r3 = r
                    break
            if r3 is None:
                return None
            prog.append(("P", 1, 3, r3))
            tgt = 1
    elif qtype in ("2in", "3in", "inp", "pin", "pni"):
        # negation types (atom-level negation, BetaE convention):
        # answers = positive-set minus negated-set(s)
        n_neg = {"2in": 1, "3in": 2, "inp": 1, "pin": 1, "pni": 1}[qtype]
        # find anchor with enough branches
        e = None
        for _ in range(80):
            r0 = rng.choice(rels_with_deg)
            cand_e = rng.choice(list(adj[r0].keys()))
            rs = [r for r in rng.sample(rels_with_deg, min(40, len(rels_with_deg)))
                  if _succ(adj, r, {cand_e})]
            if len(rs) >= 2 + n_neg:
                e = cand_e
                break
        if e is None:
            return None
        rs_all = [r for r in rng.sample(rels_with_deg, min(60, len(rels_with_deg)))
                  if _succ(adj, r, {e})]
        if len(rs_all) < 2 + n_neg:
            return None
        r_pos = rs_all[0]
        pos_set = _succ(adj, r_pos, {e})
        # pick negated relations whose successor sets are small enough
        neg_rels = []
        for r in rs_all[1:]:
            if len(neg_rels) == n_neg:
                break
            s = _succ(adj, r, {e})
            if s and len(s) <= 400:
                neg_rels.append((r, s))
        if len(neg_rels) < n_neg:
            return None
        universe = set(range(nent))
        if qtype in ("2in", "3in"):
            prog = [("A", 0, e), ("P", 10, 0, r_pos)]
            vs = [10]
            for i, (r, _) in enumerate(neg_rels):
                prog.append(("P", 20 + i, 0, r))
                prog.append(("N", 30 + i, 20 + i))
                vs.append(30 + i)
            prog.append(("I", 1, vs))
        elif qtype == "inp":
            # (pos e1 -r1-> a) ∩ ¬(e2 -r2-> b), then -r3-> t
            r2, _ = neg_rels[0]
            heads2 = list(adj[r2].keys())
            if not heads2:
                return None
            e2 = rng.choice(heads2)
            neg_set = _succ(adj, r2, {e2})
            common = pos_set - neg_set
            if not common:
                return None
            r3 = None
            for r in rng.sample(rels_with_deg, len(rels_with_deg)):
                if _succ(adj, r, common):
                    r3 = r
                    break
            if r3 is None:
                return None
            prog = [("A", 0, e), ("P", 10, 0, r_pos), ("A", 2, e2), ("P", 21, 2, r2),
                    ("N", 22, 21), ("I", 3, [10, 22]), ("P", 1, 3, r3)]
        elif qtype == "pin":
            # (e -r1-> x -r2-> A) ∩ ¬(e -r1-> x -r3-> B)
            mids = pos_set
            if not mids:
                return None
            r2 = r3 = None
            for r in rng.sample(rels_with_deg, len(rels_with_deg)):
                if _succ(adj, r, mids):
                    if r2 is None:
                        r2 = r
                    elif r3 is None and r != r2:
                        r3 = r
                        break
            if r2 is None or r3 is None:
                return None
            prog = [("A", 0, e), ("P", 1, 0, r_pos), ("P", 10, 1, r2),
                    ("P", 21, 1, r3), ("N", 22, 21), ("I", 2, [10, 22])]
        else:  # pni: projected-negated branch
            mids = pos_set
            if not mids:
                return None
            r2 = r3 = None
            for r in rng.sample(rels_with_deg, len(rels_with_deg)):
                if _succ(adj, r, mids):
                    if r2 is None:
                        r2 = r
                    elif r3 is None and r != r2:
                        r3 = r
                        break
            if r2 is None or r3 is None:
                return None
            prog = [("A", 0, e), ("P", 1, 0, r_pos), ("P", 10, 1, r2),
                    ("P", 21, 1, r3), ("N", 22, 10), ("I", 2, [22, 21])]
        tgt = prog[-1][1]
    else:
        raise ValueError(qtype)

    universe = set(range(graph.num_ent))
    env = _run_program(prog, adj, universe)
    answers = env.get(tgt, set())
    if not answers or len(answers) > max_answers:
        return None
    if anchors and (answers & set(anchors)):
        answers = answers - set(anchors)
        if not answers:
            return None
    return prog, anchors, answers


def gen_pool(graph, qtype, n, rng, hard=False, max_tries=None):
    """Generate n queries. If hard=True (eval protocol), require >=1 hard answer
    (answer reachable only via valid/test edges); returns list of dicts with
    easy/hard answer sets."""
    out = []
    tries = 0
    max_tries = max_tries or n * 400
    while len(out) < n and tries < max_tries:
        tries += 1
        res = gen_query(graph, qtype, rng)
        if res is None:
            continue
        prog, anchors, answers = res
        item = {"prog": prog, "anchors": anchors, "type": qtype}
        if hard:
            universe = set(range(graph.num_ent))
            env_full = _run_program(prog, graph.adj_full, universe)
            tgt = prog[-1][1]
            full = env_full.get(tgt, set()) - set(anchors)
            easy = answers
            hard_ans = full - easy
            if not easy and not hard_ans:
                continue
            if not hard_ans:
                continue
            item["easy"] = easy
            item["hard"] = hard_ans
            item["filter"] = full
        else:
            item["answers"] = answers
        out.append(item)
    return out
