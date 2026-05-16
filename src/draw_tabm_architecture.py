"""
draw_tabm_architecture.py
-------------------------
Renders a publication-style TabM architecture diagram (matplotlib) and saves PNG to results/.

Run from project root:
    python -m src.draw_tabm_architecture

Or:
    python src/draw_tabm_architecture.py
"""

from __future__ import annotations

import os
import sys

import matplotlib as mpl

mpl.use("Agg")  # non-interactive save (headless / CI safe)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _rounded_box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    fontsize=9,
    fc="#E8F1FB",
    ec="#2C5282",
    linewidth=1.4,
    text_color="#1A365D",
    style="round,pad=0.02",
):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=style,
        linewidth=linewidth,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=0.35,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + width / 2, xy[1] + height / 2
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight="medium",
        linespacing=1.12,
    )
    return box


def _arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    text=None,
    *,
    label_dx=0.28,
    label_dy=0.0,
    linewidth=1.6,
    color="#4A5568",
    linestyle="solid",
    mutation_scale=14,
):
    """Vertical-ish arrow; optional label to the right of the midpoint with a white backing."""
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arr)
    if text:
        mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy
        ax.text(
            mx,
            my,
            text,
            ha="left",
            va="center",
            fontsize=7.5,
            color="#2D3748",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#CBD5E0",
                linewidth=0.6,
                alpha=0.94,
            ),
        )


def draw_tabm_architecture(
    out_path: str,
    *,
    dpi: int = 300,
    fig_w: float = 11.75,
    fig_h: float = 11.5,
    k: int = 16,
    n_features: int = 122,
    n_classes: int = 5,
    n_blocks: int = 3,
    d_block: int = 256,
) -> None:
    """Draw TabM data-flow diagram and save to ``out_path``."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    # Tighten view window to reduce outer whitespace (left is fixed; top is set later
    # once element positions are known).
    ax.set_xlim(0.9, 14.2)
    ax.set_ylim(0.4, 12.3)
    ax.axis("off")

    # Diagram center: close to the left shape-key column (small horizontal gap)
    cx = 7.95
    bw, bh = 5.85, 0.72
    bh_pre = 0.88
    # Reduce vertical whitespace while keeping arrow labels readable.
    v_gap = 0.42

    # Extra padding on the right keeps "$k$ ensemble members" inside the dashed box
    pad_x_left, pad_x_right = 0.5, 1.28
    pad_y = 0.38
    block_h = 0.62
    blk_gap = 0.22
    total_inner_h = n_blocks * block_h + (n_blocks - 1) * blk_gap
    group_w = bw + pad_x_left + pad_x_right
    group_h = total_inner_h + 2 * pad_y + 0.55

    # Stack from bottom (y = lower edge of each box)
    # Start lower and pack the stack tighter (bbox_inches='tight' will crop to content).
    y_out = 0.95
    y_mean = y_out + bh + v_gap
    y_head = y_mean + bh + v_gap
    gy0 = y_head + bh + v_gap
    y_in = gy0 + group_h + v_gap
    y_pre = y_in + bh + v_gap

    # Reduce top whitespace by tightening the y-extent around the content.
    y_min = y_out - 0.35
    y_max = (y_pre + bh_pre) + 0.22
    ax.set_ylim(y_min, y_max)

    gx0 = cx - group_w / 2

    backbone = FancyBboxPatch(
        (gx0, gy0),
        group_w,
        group_h,
        boxstyle="round,pad=0.03",
        linewidth=2.0,
        edgecolor="#B7791F",
        facecolor="#FFFAF0",
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.add_patch(backbone)

    # Top-left inside dashed box — avoids sitting on Block 1 (top block)
    ax.text(
        gx0 + 0.16,
        gy0 + group_h - 0.12,
        "Backbone (BatchEnsemble)",
        ha="left",
        va="top",
        fontsize=7.8,
        color="#975A16",
        weight="bold",
    )

    bx0 = cx - bw / 2
    by_base = gy0 + pad_y + 0.52
    block_colors = ("#F7FAFC", "#EDF2F7", "#E2E8F0")
    for i in range(n_blocks):
        block_num = i + 1
        by = by_base + (n_blocks - block_num) * (block_h + blk_gap)
        fc = block_colors[i % len(block_colors)]
        _rounded_box(
            ax,
            (bx0, by),
            bw,
            block_h,
            f"Block {block_num}  ·  Linear {d_block}  ·  ReLU  ·  Dropout 0.1",
            fontsize=8,
            fc=fc,
            ec="#4A5568",
            linewidth=1.2,
        )
        # $k$ = BatchEnsemble width (# ensemble members per block); experiments use $k=16$.
        k_x = gx0 + group_w - 0.05
        ax.text(
            k_x,
            by + block_h / 2,
            rf"$k={k}$" + "\nensemble members",
            ha="right",
            va="center",
            fontsize=6.35,
            color="#718096",
            linespacing=1.05,
        )

    # --- Preprocessing (shared with baselines in this project) ---
    pre_txt = (
        "Preprocessing (shared)\n"
        r"One-hot encode 3 categorical fields · $\mathrm{StandardScaler}$ on train" + "\n"
        rf"$\rightarrow$ {n_features}-D standardised vector"
    )
    _rounded_box(
        ax,
        (cx - bw / 2, y_pre),
        bw,
        bh_pre,
        pre_txt,
        fontsize=7.8,
        fc="#F0FFF4",
        ec="#276749",
        text_color="#22543D",
    )

    _rounded_box(
        ax,
        (cx - bw / 2, y_in),
        bw,
        bh,
        f"Model input (batched)\n{n_features} numeric features",
        fc="#EBF8FF",
        ec="#2B6CB0",
    )

    _rounded_box(
        ax,
        (cx - bw / 2, y_head),
        bw,
        bh,
        f"Classification head\n$k$ × ({n_classes} logits each)",
        fc="#E6FFFA",
        ec="#2C7A7B",
    )

    _rounded_box(
        ax,
        (cx - bw / 2, y_mean),
        bw,
        bh,
        r"Mean over ensemble axis $k$",
        fc="#FAF5FF",
        ec="#6B46C1",
    )

    _rounded_box(
        ax,
        (cx - bw / 2, y_out),
        bw,
        bh,
        f"Predicted class (argmax)\n{n_classes}-way intrusion type",
        fc="#FFF5F5",
        ec="#C53030",
    )

    # Arrows top → bottom (decreasing y at tip)
    _arrow(
        ax,
        cx,
        y_pre,
        cx,
        y_in + bh,
        rf"$(B,{n_features})$ float",
        label_dx=0.32,
        label_dy=0.1,
    )
    _arrow(
        ax,
        cx,
        y_in,
        cx,
        gy0 + group_h,
        rf"$(B,{n_features})$",
        label_dx=0.32,
        label_dy=0.1,
    )
    _arrow(
        ax,
        cx,
        gy0,
        cx,
        y_head + bh,
        rf"$(B,k,{d_block})$",
        label_dx=0.32,
        label_dy=0.14,
    )
    _arrow(
        ax,
        cx,
        y_head,
        cx,
        y_mean + bh,
        rf"$(B,k,{n_classes})$",
        label_dx=0.32,
        label_dy=0.14,
    )
    _arrow(
        ax,
        cx,
        y_mean,
        cx,
        y_out + bh,
        rf"$(B,{n_classes})$",
        label_dx=0.32,
        label_dy=0.14,
    )

    # Loss is computed on averaged logits (same tensor as used for argmax at inference).
    y_mid_avg = 0.5 * (y_mean + y_out + bh)
    loss_w, loss_h = 3.25, 1.12
    # Keep the callout inside the visible x-limits (and away from tight bbox clipping).
    x_min, x_max = ax.get_xlim()
    preferred_loss_x = cx + bw / 2 + 0.32
    loss_x = min(preferred_loss_x, x_max - loss_w - 0.35)
    loss_y = y_mid_avg - loss_h / 2
    loss_box = FancyBboxPatch(
        (loss_x, loss_y),
        loss_w,
        loss_h,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        edgecolor="#C05621",
        facecolor="white",
        linestyle=(0, (5, 3)),
        zorder=5,
    )
    ax.add_patch(loss_box)
    ax.text(
        loss_x + loss_w / 2,
        loss_y + loss_h / 2 + 0.12,
        "Training loss",
        ha="center",
        va="center",
        fontsize=8.0,
        weight="bold",
        color="#9C4221",
        zorder=6,
    )
    ax.text(
        loss_x + loss_w / 2,
        loss_y + loss_h / 2 - 0.18,
        (
            rf"Weighted CE on $(B,{n_classes})$ logits vs.\ labels $y$"
            "\n"
            r"(balanced class weights; cap $50\times$ min weight)"
        ),
        ha="center",
        va="center",
        fontsize=7.0,
        color="#744210",
        linespacing=1.12,
        zorder=6,
    )
    # Tie-in to the averaged-logit stage (same arrow as argmax): dashed arrow into the callout.
    join_x0 = cx + bw / 2 + 0.08
    join_x1 = loss_x - 0.06
    _arrow(
        ax,
        join_x0,
        y_mid_avg,
        join_x1,
        y_mid_avg,
        text=None,
        linewidth=1.4,
        color="#C05621",
        linestyle=(0, (4, 2)),
        mutation_scale=12,
    )

    # Shape key sits just left of the dashed backbone (small gap)
    leg_x = 1.45
    leg_fs = 6.85
    y_leg_top = gy0 + group_h - 0.12
    lines = [
        r"$B$ — batch size",
        rf"$k$ — ensemble width ({k} ensemble members)",
        rf"$(B,{n_features})$ — standardised input",
        rf"$(B,k,{d_block})$ — hidden (per member)",
        rf"$(B,k,{n_classes})$ — logits (pre-mean)",
        rf"$(B,{n_classes})$ — averaged logits",
    ]
    ly = y_leg_top
    for line in lines:
        ax.text(
            leg_x,
            ly,
            "• " + line,
            fontsize=leg_fs,
            color="#4A5568",
            va="top",
        )
        ly -= 0.30

    # Avoid tight_layout adding extra outer padding; we want a tight crop.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def main() -> None:
    root = _project_root()
    os.chdir(root)
    out = os.path.join(root, "results", "tabm_architecture.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw_tabm_architecture(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
