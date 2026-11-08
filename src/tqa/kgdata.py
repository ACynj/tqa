"""KG loading utilities for TQA experiments.

Loads raw triple files, adds inverse relations (vocabulary-free convention),
builds torch edge tensors and the 4-type relation interaction graph
(h2h, h2t, t2h, t2t) used by the vocabulary-free relation encoder.
"""
import os
import numpy as np
import torch

RAW_DIRS = {
    "fb15k237": "/kg-datasets/fb15k237/FB15k-237/raw",
    "wn18rr": "/kg-datasets/wn18rr/raw",
    "codex-m": "/kg-datasets/codex-m/raw",
}


def load_triples(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                parts = line.rstrip("\n").split(" ")
            if len(parts) != 3:
                continue
            out.append(tuple(parts))
    return out


class GraphData:
    """One KG with inverse relations. num_rel counts forward relations;
    edge relation ids in [0, 2*num_rel): forward r, inverse r + num_rel."""

    def __init__(self, name, raw_dir):
        self.name = name
        train = load_triples(os.path.join(raw_dir, "train.txt"))
        valid = load_triples(os.path.join(raw_dir, "valid.txt"))
        test = load_triples(os.path.join(raw_dir, "test.txt"))

        self.entities = sorted({e for t in train + valid + test for e in (t[0], t[2])})
        self.relations = sorted({t[1] for t in train + valid + test})
        self.ent2id = {e: i for i, e in enumerate(self.entities)}
        self.rel2id = {r: i for i, r in enumerate(self.relations)}
        self.num_ent = len(self.entities)
        self.num_rel = len(self.relations)

        self.split_triples = {
            "train": self._to_ids(train),
            "valid": self._to_ids(valid),
            "test": self._to_ids(test),
        }
        full = np.concatenate([self.split_triples[s] for s in ("train", "valid", "test")])

        # adjacency with inverse relations, per split (train used for symbolic GT;
        # full used for hard answers)
        self.adj_train = self._build_adj(self.split_triples["train"])
        self.adj_full = self._build_adj(full)

        # torch tensors on the full graph (message passing uses observed train
        # edges at train time; eval may use train+valid+test as observed graph,
        # standard filtered protocol keeps inference graph = train edges for
        # inductive honesty; here transductive: use train edges as observed)
        self.edge_index, self.edge_rel = self._edges_torch(self.split_triples["train"])
        self.rel_graph = self._build_relation_graph(self.split_triples["train"])
        self._rel2edges_cache = None

    def rel2edges(self, device):
        """dict rel_id -> (2, k) edge tensor, cached per device."""
        if self._rel2edges_cache is not None:
            return self._rel2edges_cache
        order = torch.argsort(self.edge_rel)
        sorted_rel = self.edge_rel[order]
        sorted_ei = self.edge_index[:, order]
        uniq, counts = torch.unique_consecutive(sorted_rel, return_counts=True)
        out = {}
        start = 0
        for r, c in zip(uniq.tolist(), counts.tolist()):
            out[r] = sorted_ei[:, start:start + c].to(device)
            start += c
        self._rel2edges_cache = out
        return out

    def _to_ids(self, triples):
        return np.array(
            [[self.ent2id[h], self.rel2id[r], self.ent2id[t]] for h, r, t in triples],
            dtype=np.int64,
        )

    def _build_adj(self, arr):
        """rel_id (incl. inverse) -> head -> set(tails). Keys are plain ints."""
        adj = [dict() for _ in range(2 * self.num_rel)]
        for h, r, t in arr.tolist():
            adj[r].setdefault(h, set()).add(t)
            adj[r + self.num_rel].setdefault(t, set()).add(h)
        return adj

    def _edges_torch(self, arr):
        heads, tails, rels = [], [], []
        for h, r, t in arr:
            heads.append(h); tails.append(t); rels.append(r)
            heads.append(t); tails.append(h); rels.append(r + self.num_rel)
        ei = torch.tensor([heads, tails], dtype=torch.long)
        er = torch.tensor(rels, dtype=torch.long)
        return ei, er

    def _build_relation_graph(self, arr):
        """4 interaction types between relation types (forward only, as in the
        ULTRA convention; interactions of inverse edges are symmetric images).
        Returns dict etype -> (src, dst, weight) numpy arrays over 2*num_rel nodes.
        """
        h2h = {}; h2t = {}; t2h = {}; t2t = {}
        ent_head_rels = [set() for _ in range(self.num_ent)]
        ent_tail_rels = [set() for _ in range(self.num_ent)]
        for h, r, t in arr:
            inv = r + self.num_rel
            ent_head_rels[h].add(r)      # h is head of r
            ent_tail_rels[t].add(r)      # t is tail of r
            ent_head_rels[t].add(inv)    # t is head of inverse r
            ent_tail_rels[h].add(inv)    # h is tail of inverse r
        def bump(d, a, b):
            d[(a, b)] = d.get((a, b), 0) + 1
        for e in range(self.num_ent):
            hs, ts = ent_head_rels[e], ent_tail_rels[e]
            for a in hs:
                for b in hs:
                    if a != b: bump(h2h, a, b)
                for b in ts:
                    if a != b: bump(h2t, a, b)
            for a in ts:
                for b in hs:
                    if a != b: bump(t2h, a, b)
                for b in ts:
                    if a != b: bump(t2t, a, b)
        out = {}
        for name, d in (("h2h", h2h), ("h2t", h2t), ("t2h", t2h), ("t2t", t2t)):
            if d:
                ks = list(d.keys())
                out[name] = (
                    np.array([k[0] for k in ks], dtype=np.int64),
                    np.array([k[1] for k in ks], dtype=np.int64),
                    np.array([np.log1p(d[k]) for k in ks], dtype=np.float32),
                )
            else:
                out[name] = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
        return out


def load_graphs(names=("fb15k237", "wn18rr", "codex-m")):
    return {n: GraphData(n, RAW_DIRS[n]) for n in names}
