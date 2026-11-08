"""TQA models.

SQA-1: vocabulary-free projection + FIXED product fuzzy operators (the
       UltraQuery-class baseline = order-0 aggregation in the TQA hierarchy).
SQA-2: same projection + LEARNABLE symmetric aggregation over witness
       evidence (score + hidden witness features per branch), implementing
       order-k witness aggregation of the theory.

Both are doubly-permutation-equivariant by construction: no entity or
relation embeddings exist anywhere; relation features come from the
interaction (relation) graph conditioned on query relations only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EDGETYPES = ("h2h", "h2t", "t2h", "t2t")


class RelationEncoder(nn.Module):
    """Query-conditioned relation features via MP over the relation graph.

    Marks all relations appearing in the query plan (with direction role)
    and runs L rounds of typed message passing.
    """

    def __init__(self, d=64, layers=3):
        super().__init__()
        self.d = d
        self.mark = nn.Parameter(torch.randn(d) * 0.1)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                et: nn.Linear(d, d, bias=False) for et in EDGETYPES
            }))
        self.gru = nn.GRUCell(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, rel_graph, num_rel2, query_rels, device):
        """rel_graph: dict et -> (src,dst,w) numpy; num_rel2 = 2*num_rel;
        query_rels: LongTensor of relations in the current query."""
        h = torch.zeros(num_rel2, self.d, device=device)
        marks = torch.zeros(num_rel2, self.d, device=device)
        marks[query_rels] = self.mark
        h = h + marks
        edges = {
            et: (
                torch.from_numpy(rel_graph[et][0]).to(device),
                torch.from_numpy(rel_graph[et][1]).to(device),
                torch.from_numpy(rel_graph[et][2]).to(device),
            )
            for et in EDGETYPES
        }
        for layer in self.layers:
            agg = torch.zeros_like(h)
            for et in EDGETYPES:
                src, dst, w = edges[et]
                if src.numel() == 0:
                    continue
                m = layer[et](h)[src] * w.unsqueeze(-1)
                agg.index_add_(0, dst, m)
            h = self.gru(h, agg)
        return torch.tanh(self.out(h))  # (2R, d)


class Projection(nn.Module):
    """One-hop fuzzy projection P_r: scores' = MP conditioned on relation r.

    NBFNet-style conditional propagation with multi-source support: input is
    a fuzzy membership vector over entities (not a single anchor).
    """

    def __init__(self, d=64, layers=2):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(1, d)
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.gru = nn.GRUCell(d, d)
        self.out = nn.Linear(d, 1)

    def forward(self, scores, rel_edges, rel_feat):
        """scores: (B, E) fuzzy memberships; rel_edges: LongTensor (2, Ek)
        of edges carrying relation rel; rel_feat: (d,) features of rel."""
        B, E = scores.shape
        dev = scores.device
        h = self.inp(scores.unsqueeze(-1))  # (B, E, d)
        src, dst = rel_edges[0], rel_edges[1]
        for lin in self.layers:
            msg_src = lin(h) * rel_feat.view(1, 1, -1)
            agg = torch.zeros_like(h)
            agg.index_add_(1, dst, msg_src[:, src, :])
            h = self.gru(h.reshape(-1, self.d), agg.reshape(-1, self.d)).reshape(B, E, self.d)
        return torch.sigmoid(self.out(h)).squeeze(-1), h  # scores', witness h


class FuzzyOps(nn.Module):
    """SQA-1: fixed product t-norm / t-conorm (order-0)."""

    def intersect(self, scores_list, feats_list, rel_feats=None):
        s = scores_list[0]
        for s2 in scores_list[1:]:
            s = s * s2
        return s, None

    def union(self, scores_list, feats_list, rel_feats=None):
        s = scores_list[0]
        for s2 in scores_list[1:]:
            s = s + s2 - s * s2
        return s, None


class LearnedOps(nn.Module):
    """SQA-2: learnable symmetric aggregation over witness evidence.

    Per entity v, each branch i supplies (s_i(v), w_i(v)) where w_i is the
    witness feature from that branch's last projection. Aggregation uses
    symmetric moments (mean, max, prod of scores) + per-branch attention --
    permutation-invariant over branches (T4), vocabulary-free.
    """

    def __init__(self, d=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.fuse = nn.Sequential(
            nn.Linear(2 * d + 3, d), nn.ReLU(), nn.Linear(d, d // 2), nn.ReLU(),
            nn.Linear(d // 2, 1),
        )
        self.pos = nn.Parameter(torch.tensor(-1.0))

    def _agg(self, scores_list, feats_list, union_mode):
        k = len(scores_list)
        B, E = scores_list[0].shape
        S = torch.stack(scores_list, dim=2)  # (B, E, k)
        if union_mode:
            S = 1.0 - S
        mean_s = S.mean(dim=2)
        max_s = S.max(dim=2).values
        prod_s = S.prod(dim=2)
        H = torch.stack(feats_list, dim=2)  # (B, E, k, d)
        phi = self.phi(torch.cat([S.unsqueeze(-1), H], dim=-1))  # (B,E,k,d)
        # symmetric attention pooling (weights depend only on branch content)
        att = torch.softmax(phi.mean(dim=-1) / 0.5, dim=2)  # (B,E,k)
        pooled = (phi * att.unsqueeze(-1)).sum(dim=2)  # (B,E,d)
        z = torch.cat([pooled, phi.max(dim=2).values,
                       mean_s.unsqueeze(-1), max_s.unsqueeze(-1), prod_s.unsqueeze(-1)], dim=-1)
        out = torch.sigmoid(self.fuse(z) + self.pos).squeeze(-1)
        if union_mode:
            out = 1.0 - out
        return out, None

    def intersect(self, scores_list, feats_list, rel_feats=None):
        return self._agg(scores_list, feats_list, union_mode=False)

    def union(self, scores_list, feats_list, rel_feats=None):
        return self._agg(scores_list, feats_list, union_mode=True)


class GatedOps(nn.Module):
    """SQA-3: learnable operators INSIDE the t-norm family.

    Aczel-Alsala parameterization: T_theta(x, y) = exp(-(((-ln x)^th + (-ln
    y)^th)^(1/th))). theta=1 recovers the product t-norm exactly. Witness
    features (order>=1 evidence) modulate per-branch exponents through a
    learned map, preserving conjunction/disjunction semantics while adapting
    sharpness to structural evidence. Fixed product ops = the theta=1 point.
    """

    def __init__(self, d=64, rel_dim=None):
        super().__init__()
        self.use_rel = rel_dim is not None
        self.gate = nn.Sequential(
            nn.Linear((rel_dim or d + 1), d), nn.ReLU(), nn.Linear(d, 1)
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))  # global theta bias
        self.eps = 1e-6
        self._theta_acc = None

    def reset_reg(self):
        self._theta_acc = None

    def reg_loss(self):
        # mean squared deviation of theta from 1 (the product-t-norm point):
        # keeps gates at the robust fuzzy prior unless evidence justifies moving
        return self._theta_acc if self._theta_acc is not None else 0.0

    def intersect(self, scores_list, feats_list, rel_feats=None):
        eps = 1e-4
        thetas = []
        for i, (s, f) in enumerate(zip(scores_list, feats_list)):
            w = (s / (s.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)  # (B,E,1)
            fsum = (f * w).sum(dim=1)  # (B,d) score-weighted witness summary
            s_max = s.max(dim=1, keepdim=True).values
            _abl = __import__('os').environ.get('GATED_ABLATE', '')
            if 'score' in _abl:
                s_max = torch.zeros_like(s_max)
            if 'witness' in _abl:
                fsum = torch.zeros_like(fsum)
            inp = [s_max, fsum]
            if self.use_rel and rel_feats is not None and rel_feats[i] is not None:
                inp.append(torch.stack(rel_feats[i]))  # (B,d) per-query relation signature  # order-2: global relation signature
            t = 0.5 + 3.5 * torch.sigmoid(self.gate(torch.cat(inp, dim=-1)) + self.gate_logit)
            thetas.append(t)  # (B,1)
            dev = ((t - 1.0) ** 2).mean()
            self._theta_acc = dev if self._theta_acc is None else self._theta_acc + dev
        # Aczel-Alsala t-norm with numerically safe clamps
        L = torch.stack(
            [(-torch.log(scores_list[i].clamp(eps, 1.0 - eps))) ** thetas[i]
             for i in range(len(scores_list))],
            dim=2)  # (B,E,k)
        Lsum = L.sum(dim=2).clamp(min=eps, max=60.0)
        out = torch.exp(-Lsum.pow(1.0 / thetas[0]))
        return out.clamp(0.0, 1.0), None

    def union(self, scores_list, feats_list, rel_feats=None):
        comp = [1.0 - s for s in scores_list]
        out, _ = self.intersect(comp, feats_list, rel_feats)
        return 1.0 - out, None


class TQA(nn.Module):
    def __init__(self, d=64, ops="learned", proj_layers=2, rel_layers=3):
        super().__init__()
        self.rel_enc = RelationEncoder(d=d, layers=rel_layers)
        self.proj = Projection(d=d, layers=proj_layers)
        if ops in ("gated", "gated2"):
            self.ops = GatedOps(d, rel_dim=(d + 1 + d) if ops == "gated2" else None)
        elif ops == "learned":
            self.ops = LearnedOps(d)
        else:
            self.ops = FuzzyOps()
        self.d = d

    def forward(self, graph, plans, device):
        """Execute a batch of same-structure query plans.

        graph: GraphData (tensors precomputed); plans: list of programs
        (same op skeleton, different relations/anchors).
        Returns target scores (B, E).
        """
        B = len(plans)
        E = graph.num_ent
        # relation features conditioned on the union of query relations
        qrels = sorted({op[3] for p in plans for op in p if op[0] == "P"})
        qrels_t = torch.tensor(qrels, dtype=torch.long, device=device)
        rel_feat = self.rel_enc(graph.rel_graph, 2 * graph.num_rel, qrels_t, device)

        # precollected edge index per relation (cached on the graph object)
        rel2edges_t = graph.rel2edges(device)

        # batched anchor scores
        anchor_scores = torch.zeros(B, E, device=device)
        for b, p in enumerate(plans):
            for op in p:
                if op[0] == "A":
                    anchor_scores[b, op[2]] = 1.0

        # op skeleton from first plan; variable-major execution
        skeleton = [(op[0], op[1], op[2] if isinstance(op[2], list) else None)
                    for op in plans[0]]
        env_s, env_f, env_r = {}, {}, {}
        # anchors
        for b, p in enumerate(plans):
            for op in p:
                if op[0] == "A":
                    cur = env_s.get(op[1])
                    if cur is None:
                        cur = torch.zeros(B, E, device=device)
                        env_s[op[1]] = cur
                    cur[b, op[2]] = 1.0
        for oi, op0 in enumerate(plans[0]):
            kind = op0[0]
            if kind == "P":
                _, out, var, _ = op0
                rels = [p[oi][3] for p in plans]  # op at the same position
                s_in = env_s[var]
                outs, hs = [], []
                for b, r in enumerate(rels):
                    ei = rel2edges_t.get(r)
                    if ei is None:
                        outs.append(torch.zeros(E, device=device))
                        hs.append(torch.zeros(E, self.d, device=device))
                        continue
                    ei = ei.to(device)
                    rf = rel_feat[r]
                    s1, h1 = self.proj(s_in[b:b + 1], ei, rf)
                    outs.append(s1[0])
                    hs.append(h1[0])
                env_s[out] = torch.stack(outs, dim=0)
                env_f[out] = torch.stack(hs, dim=0)
                env_r[out] = [rel_feat[r] for r in rels]
            elif kind == "I":
                _, out, vs = op0
                ss = [env_s[v] for v in vs]
                ff = [env_f.get(v) for v in vs]
                if any(f is None for f in ff):
                    ff = [s.unsqueeze(-1).expand(-1, -1, self.d) for s in ss]
                env_s[out], _ = self.ops.intersect(ss, ff, [env_r.get(v) for v in vs])
                env_f.pop(out, None)
            elif kind == "U":
                _, out, vs = op0
                ss = [env_s[v] for v in vs]
                ff = [env_f.get(v) for v in vs]
                if any(f is None for f in ff):
                    ff = [s.unsqueeze(-1).expand(-1, -1, self.d) for s in ss]
                env_s[out], _ = self.ops.union(ss, ff, [env_r.get(v) for v in vs])
                env_f.pop(out, None)
            elif kind == "N":
                # atom negation: complement (non-parametric, all classes)
                _, out, var = op0
                env_s[out] = 1.0 - env_s[var]
                if var in env_f:
                    env_f[out] = env_f[var]
        tgt = plans[0][-1][1]
        return env_s[tgt]
