#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASIC_METRICS = ["RMSE", "MAE", "PBIAS_%", "Pearson_r"]
PRECIP_METRICS = ["CSI_0.1", "CSI_1.0", "CSI_5.0", "CSI_10.0", "FSS_w3", "FSS_w5", "FSS_w9"]
LOWER_BETTER = {"RMSE", "MAE", "PBIAS_%"}
HIGHER_BETTER = {"Pearson_r", "CSI_0.1", "CSI_1.0", "CSI_5.0", "CSI_10.0", "FSS_w3", "FSS_w5", "FSS_w9"}

# Canonical names -> aliases across runs
ALIASES = {
    "PRECIP": ["APCPsurface", "APCP_surface", "RAIN_TOTAL"],
    "DLWRF_GLW": ["DLWRFsurface", "DLWRF_surface", "GLW"],
    "DSWRF_SWDOWN": ["DSWRFsurface", "DSWRF_surface", "SWDOWN"],
    "PRES_PSFC": ["PRESsurface", "PRES_surface", "PSFC"],
    "SPFH_Q2": ["SPFH2maboveground", "SPFH_2maboveground", "Q2"],
    "TMP_T2": ["TMP2maboveground", "TMP_2maboveground", "T2"],
    "UGRD_U10": ["UGRD10maboveground", "UGRD_10maboveground", "U10"],
    "VGRD_V10": ["VGRD10maboveground", "VGRD_10maboveground", "V10"],
}


def build_alias_lookup():
    out = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            out[alias] = canonical
    return out


def normalize_df(df):
    if "variable" not in df.columns:
        raise ValueError("Expected 'variable' column in metrics table")
    alias_lookup = build_alias_lookup()
    df = df.copy()
    df["canonical_variable"] = df["variable"].map(alias_lookup).fillna(df["variable"])
    return df


def percent_skill(metric, baseline, candidate):
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return np.nan

    if metric in LOWER_BETTER:
        b = abs(baseline) if metric == "PBIAS_%" else baseline
        c = abs(candidate) if metric == "PBIAS_%" else candidate
        denom = max(abs(b), 1e-12)
        return 100.0 * (b - c) / denom

    if metric in HIGHER_BETTER:
        denom = max(abs(1.0 - baseline), 1e-12)
        return 100.0 * (candidate - baseline) / denom

    return np.nan


def merge_metric_tables(transfer_df, wrf_df, aorc_df, metrics, value_label):
    t = normalize_df(transfer_df)
    w = normalize_df(wrf_df)
    a = normalize_df(aorc_df)

    keep_cols = ["canonical_variable"] + [m for m in metrics if m in t.columns or m in w.columns or m in a.columns]

    t = t[[c for c in keep_cols if c in t.columns]].copy()
    w = w[[c for c in keep_cols if c in w.columns]].copy()
    a = a[[c for c in keep_cols if c in a.columns]].copy()

    for m in metrics:
        if m not in t.columns:
            t[m] = np.nan
        if m not in w.columns:
            w[m] = np.nan
        if m not in a.columns:
            a[m] = np.nan

    t = t[["canonical_variable"] + metrics].rename(columns={m: f"transfer_{m}" for m in metrics})
    w = w[["canonical_variable"] + metrics].rename(columns={m: f"wrf3km_{m}" for m in metrics})
    a = a[["canonical_variable"] + metrics].rename(columns={m: f"aorc3km_{m}" for m in metrics})

    out = t.merge(w, on="canonical_variable", how="outer").merge(a, on="canonical_variable", how="outer")

    for m in metrics:
        out[f"transfer_vs_wrf3km_skill_%_{m}"] = out.apply(
            lambda r: percent_skill(m, r.get(f"wrf3km_{m}", np.nan), r.get(f"transfer_{m}", np.nan)), axis=1
        )
        out[f"transfer_vs_aorc3km_skill_%_{m}"] = out.apply(
            lambda r: percent_skill(m, r.get(f"aorc3km_{m}", np.nan), r.get(f"transfer_{m}", np.nan)), axis=1
        )

    out = out.sort_values("canonical_variable").reset_index(drop=True)
    out.insert(0, "table", value_label)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Compare transfer run metrics against WRF3km and AORC3km.")
    p.add_argument("--transfer-basic", type=Path, required=True)
    p.add_argument("--transfer-precip", type=Path, required=True)
    p.add_argument("--wrf-basic", type=Path, required=True)
    p.add_argument("--wrf-precip", type=Path, required=True)
    p.add_argument("--aorc-basic", type=Path, required=True)
    p.add_argument("--aorc-precip", type=Path, required=True)
    p.add_argument("--out-basic", type=Path, required=True)
    p.add_argument("--out-precip", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    transfer_basic = pd.read_csv(args.transfer_basic)
    transfer_precip = pd.read_csv(args.transfer_precip)
    wrf_basic = pd.read_csv(args.wrf_basic)
    wrf_precip = pd.read_csv(args.wrf_precip)
    aorc_basic = pd.read_csv(args.aorc_basic)
    aorc_precip = pd.read_csv(args.aorc_precip)

    basic_out = merge_metric_tables(
        transfer_df=transfer_basic,
        wrf_df=wrf_basic,
        aorc_df=aorc_basic,
        metrics=BASIC_METRICS,
        value_label="basic",
    ).round(4)

    precip_out = merge_metric_tables(
        transfer_df=transfer_precip,
        wrf_df=wrf_precip,
        aorc_df=aorc_precip,
        metrics=PRECIP_METRICS,
        value_label="precip",
    ).round(4)

    basic_out.to_csv(args.out_basic, index=False)
    precip_out.to_csv(args.out_precip, index=False)

    print(f"Saved: {args.out_basic}")
    print(f"Saved: {args.out_precip}")
