"""Reconstruct test_answers_easy.pkl for FB-122 (lost to a truncated zip).

easy = exact set-semantics evaluation of each test query on the VALID graph
(train + val_inference).  Self-check: evaluating on the TEST graph
(train + test_inference) must reproduce the surviving test_answers_hard.pkl.
"""
import pickle
import sys
from collections import defaultdict

D = "/root/git/ULTRA/query-datasets/122"
NUM_ENT = 14444          # stats.txt
UNIV = frozenset(range(NUM_ENT))


def load_adj(*files):
    adj = defaultdict(lambda: defaultdict(set))
    for f in files:
        for line in open(f"{D}/{f}"):
            h, r, t = map(int, line.split())
            adj[r][h].add(t)
    return {r: dict(v) for r, v in adj.items()}


adj_valid = load_adj("train_graph.txt", "val_inference.txt")
adj_test = load_adj("train_graph.txt", "test_inference.txt")
queries = pickle.load(open(f"{D}/test_queries.pkl", "rb"))
hard = pickle.load(open(f"{D}/test_answers_hard.pkl", "rb"))


def to_prog(qq):
    """Copy of tqa.eval_zero.to_prog (struct -> A/P/I/U/N program)."""
    prog = []
    ctr = [100]
    def newv():
        ctr[0] += 1
        return ctr[0]
    def parse(node):
        if isinstance(node[0], int) and node[0] >= 0 and isinstance(node[1], tuple):
            e, chain = node
            v = newv()
            prog.append(("A", v, e))
            for step in chain:
                if step == "n" or step == -2:
                    vn = newv(); prog.append(("N", vn, v)); v = vn
                else:
                    vn = newv(); prog.append(("P", vn, v, step)); v = vn
            return v
        subs = []
        is_union = False
        for s in node:
            if isinstance(s, tuple) and len(s) == 1 and s[0] in (-1, "u"):
                is_union = True
            elif isinstance(s, tuple) and len(s) == 1 and s[0] in ("n", -2):
                continue
            else:
                subs.append(parse(s))
        out = newv()
        prog.append(("U" if is_union else "I", out, subs))
        return out
    def top(node):
        if isinstance(node[0], int) and node[0] >= 0:
            return parse(node)
        if len(node) == 2 and isinstance(node[1], tuple) and node[1] and \
                all(isinstance(x, int) and x >= 0 for x in node[1]):
            base = parse(node[0]); v = base
            for r in node[1]:
                vn = newv(); prog.append(("P", vn, v, r)); v = vn
            return v
        return parse(node)
    tgt = top(qq)
    return prog, tgt


def run(prog, tgt, adj):
    env = {}
    for op in prog:
        if op[0] == "A":
            env[op[1]] = {op[2]}
        elif op[0] == "P":
            _, vn, v, r = op
            tails = adj.get(r, {})
            env[vn] = set().union(*(tails.get(h, set()) for h in env[v])) if env[v] else set()
        elif op[0] == "N":
            env[op[1]] = UNIV - env[op[2]]
        elif op[0] == "I":
            subs = [env[s] for s in op[2]]
            out = subs[0]
            for s in subs[1:]:
                out = out & s
            env[op[1]] = out
        elif op[0] == "U":
            out = set()
            for s in op[2]:
                out |= env[s]
            env[op[1]] = out
    return env[tgt]


easy = defaultdict(dict)
n_check = n_match = n_q = 0
mismatch = []
for struct, qs in queries.items():
    for qq in qs:
        prog, tgt = to_prog(qq)
        easy[struct][qq] = run(prog, tgt, adj_valid)
        n_q += 1
        ha = hard.get(struct, {}).get(qq)
        if ha is not None:
            n_check += 1
            if run(prog, tgt, adj_test) == set(ha):
                n_match += 1
            elif len(mismatch) < 5:
                mismatch.append((struct, qq))

print(f"queries={n_q}  hard-check: {n_match}/{n_check} exact "
      f"({100 * n_match / max(n_check, 1):.2f}%)")
if mismatch:
    print("sample mismatches:", mismatch)
    sys.exit(1)

pickle.dump(dict(easy), open(f"{D}/test_answers_easy.pkl", "wb"))
sizes = {s: len(v) for s, v in easy.items()}
print("saved easy:", sizes)
