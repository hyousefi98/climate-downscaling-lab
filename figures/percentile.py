"""Plot the revised percentile-FSS comparison from external fss.csv/audit.json."""
import argparse
import csv
import json
from pathlib import Path


def draw(csv_path,audit_path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    with Path(csv_path).open() as f:rows=list(csv.DictReader(f))
    audit=json.loads(Path(audit_path).read_text())
    quantiles=audit['predictor_quantiles']
    if not quantiles:raise ValueError('No percentile definitions')
    fig,axes=plt.subplots(1,len(quantiles),figsize=(4.4*len(quantiles),3.8),sharey=True,squeeze=False,constrained_layout=True)
    for ax,pq,rq in zip(axes[0],quantiles,audit['reference_quantiles']):
        key=f"quantile_{pq['quantile']:g}"
        selected=[r for r in rows if r['experiment']==key]
        if not selected:raise ValueError(f'No FSS rows for {key}')
        selected.sort(key=lambda r:int(r['window_cells']))
        ax.plot([int(r['window_cells']) for r in selected],[float(r['FSS']) if r['FSS'] else float('nan') for r in selected],marker='o',label='Product-specific percentile')
        for name in dict.fromkeys(r['experiment'] for r in rows if r['experiment'].startswith('absolute_')):
            a=sorted([r for r in rows if r['experiment']==name],key=lambda r:int(r['window_cells']))
            ax.plot([int(r['window_cells']) for r in a],[float(r['FSS']) if r['FSS'] else float('nan') for r in a],ls='--',label=name.replace('absolute_',''))
        ax.set(title=f"q{100*pq['quantile']:g}: predictor {pq['threshold_mm']:.3f} mm\nreference {rq['threshold_mm']:.3f} mm",xlabel='Neighborhood width (grid cells)',ylim=(0,1))
        ax.grid(axis='y',alpha=.25)
    axes[0,0].set_ylabel('Fractions Skill Score');axes[0,0].legend(frameon=False)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=300);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();draw(a.csv,a.audit,a.output)


if __name__=='__main__':main()
