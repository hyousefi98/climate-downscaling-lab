"""Plot externally supplied metric tables; never embeds paper results.

python -m figures.plot --csv fss.csv --x window --y fss --hue model --output fss.pdf
"""
import argparse
from pathlib import Path


def plot_table(csv_path, output, x, y, hue, *, xlabel=None, ylabel=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    data = pd.read_csv(csv_path)
    for name in (x, y, hue):
        if name not in data:
            raise ValueError(f"Missing column {name}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    for name, group in data.groupby(hue, sort=False):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", label=name)
    ax.set(xlabel=xlabel or x, ylabel=ylabel or y)
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--hue", required=True)
    p.add_argument("--xlabel")
    p.add_argument("--ylabel")
    a=p.parse_args()
    plot_table(a.csv,a.output,a.x,a.y,a.hue,xlabel=a.xlabel,ylabel=a.ylabel)


if __name__=="__main__":
    main()
