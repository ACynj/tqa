"""Shared figure helpers implementing figures4papers api.md for TQA paper.
Implements: PALETTE, FigureStyle, apply_publication_style, create_subplots,
finalize_figure, make_grouped_bar, annotate_bars (per api.md)."""
from dataclasses import dataclass

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E",
    "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                  PALETTE["teal"], PALETTE["violet"]]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style=None):
    import matplotlib.pyplot as plt
    st = style or FigureStyle()
    plt.rcParams.update({
        "font.family": list(st.font_family), "font.size": st.font_size,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": st.axes_linewidth,
        "text.usetex": st.use_tex,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "xtick.major.width": st.axes_linewidth * 0.6,
        "ytick.major.width": st.axes_linewidth * 0.6,
        "legend.frameon": False,
    })


def create_subplots(nrows=1, ncols=1, figsize=None, **kw):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    import numpy as _np
    return fig, (_np.asarray(axes).ravel())


def make_grouped_bar(ax, categories, series, labels, ylabel="Value",
                     colors=None, annotate=False, bar_width=0.35):
    colors = colors or DEFAULT_COLORS[:len(series)]
    import numpy as np
    x = np.arange(len(categories))
    handles = []
    for k, (vals, lab, c) in enumerate(zip(series, labels, colors)):
        off = (k - (len(series) - 1) / 2) * bar_width
        h = ax.bar(x + off, vals, bar_width * 0.92, label=lab, color=c,
                   ec="white", lw=0.6)
        handles.append(h)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if annotate:
        for h in handles:
            annotate_bars(ax, h)
    return handles[-1] if handles else None


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=10, padding=3):
    import numpy as np
    ymax = ax.get_ylim()[1]
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + ymax * 0.012,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize)


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kw):
    import os
    from pathlib import Path
    if formats is None:
        formats = ["pdf", "png"]
    outs = []
    for f in formats:
        p = Path(f"{out_path}.{f}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kw)
        outs.append(p)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return outs
