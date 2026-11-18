"""Locate the paper generators; supply paths through the original CLI options.

python -m figures.paper 2 -- --help
python -m figures.paper fss-wrf -- --diff-file DIFF.nc --reg-file REG.nc ...

Figure 1 has a reconstructed source: python -m figures.design --output design.pdf.
S3 in the original submission duplicated S2 despite an ERA5 caption; S3 here selects
the ERA5 spectra generator explicitly, rather than claiming an original exact match.
The original scripts can have inherited WRF defaults even in the ERA5 directory;
always provide --diff-file/--reg-file/--landmask-path (or documented CSV option).
"""
import argparse
from pathlib import Path
import subprocess
import sys

GENERATORS={
    "2":"wrf/plot_selected_times_paper_grid.py",
    "3":"wrf/paper/plot_fss_journal.py",
    "4":"wrf/paper/plot_fss_journal.py",
    "5":"era5_conus404/plot_selected_times_paper_grid.py",
    "6":"era5_conus404/paper/code/plot_qq_q95_spectra_journal.py",
    "S1":"wrf/inference/plot_single_sample_wrf_time.py",
    "S2":"wrf/paper/plot_spectra_all_variables_journal.py",
    "S3":"era5_conus404/paper/code/plot_spectra_all_variables_journal.py",
    "S4":"era5_conus404/inference/plot_single_sample_wrf_time.py",
    "fss-wrf":"wrf/plot_precip_fss_prob.py",
    "fss-era5":"era5_conus404/plot_precip_fss_prob_conus404.py",
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("figure",choices=GENERATORS)
    p.add_argument("arguments",nargs=argparse.REMAINDER)
    a=p.parse_args();args=a.arguments[1:] if a.arguments[:1]==["--"] else a.arguments
    path=Path(__file__).resolve().parent/"historical"/GENERATORS[a.figure]
    raise SystemExit(subprocess.call([sys.executable,str(path),*args]))


if __name__=="__main__":main()
