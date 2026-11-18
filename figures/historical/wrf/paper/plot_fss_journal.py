#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CSV_DEFAULT = Path("plots_fss_prob_renamed_labels/fss_vs_scale_values.csv")
OUT_DIR_DEFAULT = Path("paper/figures")

METHOD_ORDER = ["Input", "UNet", "CorrDiff (ensemble prob)"]
METHOD_COLORS = {
    "Input": "#1f77b4",
    "UNet": "#d62728",
    "CorrDiff (ensemble prob)": "#2ca02c",
}
METHOD_MARKERS = {
    "Input": "o",
    "UNet": "s",
    "CorrDiff (ensemble prob)": "^",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replot FSS-vs-scale in journal style from existing CSV values."
    )
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0])
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--stem", type=str, default="fss_vs_spatial_scale_journal")
    # Horizontal FSS=0.5 "skillful" reference line (on by default for backward compat).
    p.add_argument("--hline", action=argparse.BooleanOptionalAction, default=True,
                   help="Draw the horizontal FSS=0.5 reference line.")
    # Optional vertical reference line at the native input grid spacing, e.g. ERA5 ~28 km.
    p.add_argument("--vline-km", type=float, default=None,
                   help="If set, draw a dashed vertical line at this scale (km), e.g. 28.")
    p.add_argument("--vline-label", type=str, default="ERA5 native (~28 km)",
                   help="Annotation text drawn next to --vline-km.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["method"] = df["method"].replace({"CorrDiff ensemble prob": "CorrDiff (ensemble prob)"})
    df = df[df["method"].isin(METHOD_ORDER)].copy()

    thr_keep = [float(v) for v in args.thresholds]
    df = df[df["threshold_mm"].isin(thr_keep)].copy()
    if df.empty:
        raise RuntimeError(f"No rows after filtering thresholds={thr_keep} from {args.csv}")

    scales = np.sort(df["scale_km"].dropna().unique().astype(float))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.808), sharex=True, sharey=True)
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for i, thr in enumerate(thr_keep):
        ax = axes[i]
        sub = df[df["threshold_mm"] == float(thr)]

        for method in METHOD_ORDER:
            cur = (
                sub[sub["method"] == method][["scale_km", "fss"]]
                .dropna()
                .sort_values("scale_km")
            )
            x = cur["scale_km"].to_numpy(dtype=float)
            y = cur["fss"].to_numpy(dtype=float)
            h = ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.5,
                markersize=3.6,
                markeredgewidth=0.0,
                label=method,
                zorder=3,
            )[0]
            if i == 0:
                legend_handles.append(h)
                legend_labels.append(method)

        if args.hline:
            ax.axhline(0.5, color="#7a7a7a", linestyle="--", linewidth=0.8, alpha=0.75, zorder=1)
        if args.vline_km is not None:
            ax.axvline(args.vline_km, color="#7a7a7a", linestyle="--", linewidth=0.9,
                       alpha=0.85, zorder=2)
        ax.grid(True, color="#c8c8c8", alpha=0.4, linewidth=0.55, zorder=0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(float(np.min(scales)) - 2.0, float(np.max(scales)) + 2.0)
        ax.set_xticks([0, 25, 50, 75, 100, 125])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(f"Threshold = {thr:g} mm", pad=6, fontsize=13, fontweight="normal")
        # Show tick-number labels on every panel even with shared axes.
        ax.tick_params(labelbottom=True, labelleft=True)

    for j in range(len(thr_keep), 4):
        axes[j].axis("off")

    axes[0].set_ylabel("FSS")
    axes[2].set_ylabel("FSS")
    axes[2].set_xlabel("Spatial scale (km)")
    axes[3].set_xlabel("Spatial scale (km)")

    if args.vline_km is not None:
        vline_proxy = Line2D(
            [0], [0], color="#7a7a7a", linestyle="--", linewidth=0.9, alpha=0.85
        )
        legend_handles.append(vline_proxy)
        legend_labels.append(args.vline_label)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=True,
        bbox_to_anchor=(0.5, -0.005),
        handlelength=2.0,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.16, wspace=0.20, hspace=0.32)

    out_png = args.out_dir / f"{args.stem}.png"
    out_pdf = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png.resolve()}")
    print(f"Saved: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
