"""Run UQ eval on FB-122 with a dense (heap-safe) build_relation_graph.

The stock torch.sparse.mm corrupts the heap for this graph's sparsity
pattern (malloc(): unaligned tcache chunk).  The relation graph only uses
the nonzero pattern of four products, so dense equivalents are exact.
"""
import runpy
import sys

import torch

sys.path.insert(0, "/mnt/ssd/tqa/external/ULTRA")

import ultra.tasks as T
from torch_geometric.data import Data


def dense_build_relation_graph(graph):
    ei, et = graph.edge_index, graph.edge_type
    n = max(int(ei.max()) + 1, int(graph.num_nodes or 0))
    r = graph.num_relations

    def mat(rows):  # binary node x rel presence matrix
        m = torch.zeros((n, r), dtype=torch.float32)
        m[rows, et] = 1.0
        return m

    h = mat(ei[0])
    t = mat(ei[1])
    hn = h / h.sum(1, keepdim=True).clamp(min=1)
    tn = t / t.sum(1, keepdim=True).clamp(min=1)

    blocks = []  # (matrix, edge-type-id)
    blocks.append((hn.T @ h, 0))   # head-head
    blocks.append((tn.T @ t, 1))   # tail-tail
    blocks.append((hn.T @ t, 2))   # head-tail
    blocks.append((tn.T @ h, 3))   # tail-head

    idx, typ = [], []
    for m, k in blocks:
        rr, cc = (m > 0).nonzero(as_tuple=True)
        idx.append(torch.vstack([rr, cc]))
        typ.append(torch.full((rr.numel(),), k, dtype=torch.long))

    graph.relation_graph = Data(
        edge_index=torch.cat(idx, dim=1),
        edge_type=torch.cat(typ),
        num_nodes=r,
        num_relations=4,
    )
    return graph


T.build_relation_graph = dense_build_relation_graph

sys.argv = ["/mnt/ssd/tqa/external/ULTRA/script/run_query.py",
            "-c", "config/ultraquery/inductive.yaml",
            "--dataset", "InductiveFB15k237Query",
            "--version", "122",
            "--epochs", "0", "--bpe", "null", "--gpus", "[0]",
            "--bs", "32", "--threshold", "0.0",
            "--ultra_ckpt", "null",
            "--qe_ckpt", "/mnt/ssd/tqa/external/ULTRA/ckpts/ultraquery.pth"]
runpy.run_path("/mnt/ssd/tqa/external/ULTRA/script/run_query.py", run_name="__main__")
