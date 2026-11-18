"""Reconstructed experiment schematic; original submitted editable source unavailable.

No measured outcomes are embedded. The common-reference transfer panel represents
the revised matched design, not the original differently referenced headline.
"""
import argparse
from pathlib import Path


def draw(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    ax.set(xlim=(0,10),ylim=(0,5));ax.axis("off")
    rows=[("I · Synthetic benchmark","Coarsened AORC","AORC 3 km"),
          ("II · Frozen-model transfer","WRF 12 km","Same AORC 3 km"),
          ("III · Paired simulations","WRF 12 km","WRF 3 km"),
          ("IV · Reanalysis downscaling","ERA5","CONUS404")]
    for i,(title,inp,target) in enumerate(rows):
        y=4.2-i*1.1
        ax.text(.1,y+.2,title,fontsize=11,va="center")
        for x,label in [(3.4,inp),(7.1,target)]:
            ax.add_patch(FancyBboxPatch((x,y-.2),2.6,.75,boxstyle="round,pad=.08",facecolor="white",edgecolor="black"))
            ax.text(x+1.3,y+.18,label,ha="center",va="center",fontsize=11)
        ax.annotate("",xy=(7,y+.18),xytext=(6.1,y+.18),arrowprops={"arrowstyle":"->"})
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=300);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    draw(p.parse_args().output)


if __name__=="__main__":main()
