"""Aggregate ALL TQA experiment results into runs/FINAL_REPORT.md."""
import glob
import json
import os

import numpy as np

R = "/mnt/ssd/tqa/runs"
EPFO = ["1p", "2p", "2u", "2i", "up", "pi", "ip", "3i"]
NEG = ["2in", "3in", "inp", "pin", "pni"]
BRANCH = {"1p": 0, "2p": 0, "2u": 1, "up": 1, "2i": 2, "pi": 2, "ip": 3, "3i": 3}
MODELS = [("fixed", "SQA-1(fixed fuzzy)"), ("learned", "SQA-2(free-form)"),
          ("gated", "SQA-3(gated t-norm)")]


def load(name):
    f = os.path.join(R, f"{name}_results.json")
    return json.load(open(f)) if os.path.exists(f) else None


def phase2_table(out):
    out.append("\n## 1. 主实验：3 模型 x 5 种子 x 50k 步（fb15k237, hard-answer MRR）\n")
    data = {}
    for ops, _ in MODELS:
        rs = [load(f"p2_{ops}_s{s}_50k") for s in (42, 43, 44, 45, 46)]
        rs = [r for r in rs if r]
        data[ops] = rs
        if not rs:
            out.append(f"- {ops}: 结果未完成")
            continue
    n_seeds = {k: len(v) for k, v in data.items()}
    out.append(f"种子数: {n_seeds}\n")
    out.append("| 查询 | 分支 | " + " | ".join(l for _, l in MODELS) + " |")
    out.append("|---|---|" + "---|" * len(MODELS))
    for qt in EPFO + NEG[:0]:
        row = [f"{qt} | {BRANCH.get(qt,'')}", qt][0:1] + [str(BRANCH.get(qt, ""))]
        cells = []
        for ops, _ in MODELS:
            if not data.get(ops):
                cells.append("-")
                continue
            vals = [r["per_type"][qt]["MRR"] for r in data[ops] if qt in r["per_type"]]
            cells.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}" if vals else "-")
        out.append(f"| {qt} | {BRANCH.get(qt,'')} | " + " | ".join(cells) + " |")
    row = "| **AVG** |  | "
    cells = []
    for ops, _ in MODELS:
        if not data.get(ops):
            cells.append("-")
            continue
        vals = [r["avg"]["MRR"] for r in data[ops]]
        cells.append(f"**{np.mean(vals):.4f}±{np.std(vals):.4f}**")
    out.append(row + " | ".join(cells) + " |")
    # paired wins gated vs fixed
    try:
        g = [r["avg"]["MRR"] for r in data["gated"]]
        f = [r["avg"]["MRR"] for r in data["fixed"]]
        pairs = [(a, b) for a, b in zip(g, f)]
        wins = sum(1 for a, b in pairs if a > b)
        out.append(f"\n配对比较（gated vs fixed，逐种子）: gated 胜 {wins}/{len(pairs)}，"
                   f"平均Δ = {np.mean([a-b for a,b in pairs]):+.4f}")
    except Exception:
        pass
    # easy reference
    try:
        for ops, label in MODELS:
            vals = [r["avg"].get("MRR_easy") for r in data[ops] if r["avg"].get("MRR_easy") is not None]
            if vals:
                out.append(f"{label} 的 easy-answer 参考MRR = {np.mean(vals):.4f}（协议对照用）")
                break
    except Exception:
        pass


def loo_table(out):
    out.append("\n## 2. 零样本跨图迁移（LOO：留一图不训练，在其上评测，2种子）\n")
    out.append("| 目标图(未见) | " + " | ".join(l for _, l in MODELS) + " |")
    out.append("|---|" + "---|" * len(MODELS))
    for ex in ("fb15k237", "wn18rr", "codex-m"):
        cells = []
        for ops, _ in MODELS:
            rs = [load(f"loo_{ops}_ex_{ex}_s{s}") for s in (42, 43)]
            rs = [r for r in rs if r]
            if len(rs) == 2:
                a, b = (r["avg"]["MRR"] for r in rs)
                cells.append(f"{(a+b)/2:.4f}±{abs(a-b)/2:.4f}")
            elif len(rs) == 1:
                cells.append(f"{rs[0]['avg']['MRR']:.4f}")
            else:
                cells.append("-")
        out.append(f"| {ex} | " + " | ".join(cells) + " |")


def negation_table(out):
    out.append("\n## 3. 否定查询（未训练过否定类型，零样本算子泛化）\n")
    files = sorted(glob.glob(os.path.join(R, "extra_*.json")))
    rows = {}
    for f in files:
        j = json.load(open(f))
        if "negation" not in j:
            continue
        rows[os.path.basename(f)[6:-5]] = j["negation"]["avg"]["MRR"]
    if rows:
        out.append("| checkpoint | 否定类平均MRR |")
        out.append("|---|---|")
        for k, v in rows.items():
            out.append(f"| {k} | {v:.4f} |")
    else:
        out.append("(未完成)")


def collision_table(out):
    out.append("\n## 4. T2 分离实验：2i 查询按分支得分余弦相似度分桶（碰撞区=高余弦）\n")
    out.append("预测：固定算子在高余弦桶（碰撞区）相对更差；门控算子差距缩小。\n")
    files = sorted(glob.glob(os.path.join(R, "extra_*.json")))
    any_row = False
    for f in files:
        j = json.load(open(f))
        if "collision_2i" not in j:
            continue
        any_row = True
        name = os.path.basename(f)[6:-5]
        out.append(f"**{name}**（ops={j['ops']}）:")
        out.append("| 桶 | cos范围 | MRR | n |")
        out.append("|---|---|---|---|")
        for b in j["collision_2i"]:
            out.append(f"| {b['bucket']} | [{b['cos_lo']},{b['cos_hi']}] | {b['MRR']:.4f} | {b['n']} |")
        out.append("")
    if not any_row:
        out.append("(未完成)")


def main():
    out = ["# TQA 实验总报告（自动生成）", ""]
    phase2_table(out)
    loo_table(out)
    negation_table(out)
    collision_table(out)
    out.append("\n## 5. 结论判据（自动对照）\n")
    out.append("- [ ] 主实验：gated 均值≥fixed 且方差更小（见§1 配对比较）")
    out.append("- [ ] 零样本：LOO 上 gated ≥ fixed（见§2）")
    out.append("- [ ] 否定泛化：未训练类上模型仍可用（见§3，绝对值参考）")
    out.append("- [ ] 分离信号：高余弦桶中 gated 优势放大（见§4）")
    print("\n".join(out))


if __name__ == "__main__":
    main()
