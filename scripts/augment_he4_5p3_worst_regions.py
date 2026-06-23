from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from generate_refprop_he4_extended import init_refprop, refprop_props


def sample_box(n, t_bounds, p_bounds, seed, log_p=True):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n)
    t = t_bounds[0] + (t_bounds[1] - t_bounds[0]) * u[:, 0]
    if log_p:
        lp = np.log10(p_bounds[0]) + (np.log10(p_bounds[1]) - np.log10(p_bounds[0])) * u[:, 1]
        p = 10.0**lp
    else:
        p = p_bounds[0] + (p_bounds[1] - p_bounds[0]) * u[:, 1]
    return t, p


def make_rows(rp, units, molar_mass, temps, pressures, label):
    rows = []
    errors = []
    for i, (t, p) in enumerate(zip(temps, pressures)):
        props, err = refprop_props(rp, units, molar_mass, t, p)
        if props is None:
            errors.append((label, float(t), float(p), str(err)))
            continue
        row = {
            "Temperature (K)": float(t),
            "Pressure (MPa)": float(p),
            "phase_region": label,
            "Psat (MPa)": np.nan,
        }
        row.update(props)
        rows.append(row)
    return rows, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/he4data_single_phase_5p3_300K.csv")
    parser.add_argument("--out", default="data/he4data_single_phase_5p3_300K_augmented.csv")
    parser.add_argument("--refprop-root", default=r"E:\REFPROP10\REFPROP")
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    rp, units, molar_mass, _, _ = init_refprop(args.refprop_root)
    regions = [
        ("critical_cp_region", 45000, (5.25, 6.2), (0.20, 0.45), False),
        ("critical_density_region", 25000, (5.25, 7.0), (0.20, 0.60), False),
        ("highT_lowP_density_region", 30000, (250.0, 300.0), (0.001, 0.004), True),
        ("lowT_lowP_density_region", 20000, (5.3, 20.0), (0.001, 0.01), True),
    ]
    all_rows = []
    all_errors = []
    for j, (label, n, tb, pb, log_p) in enumerate(regions):
        temps, pressures = sample_box(n, tb, pb, args.seed + j, log_p=log_p)
        rows, errors = make_rows(rp, units, molar_mass, temps, pressures, label)
        all_rows.extend(rows)
        all_errors.extend(errors)
        print(label, "requested", n, "accepted", len(rows), "errors", len(errors))

    base = pd.read_csv(args.base)
    aug = pd.DataFrame(all_rows)
    out = pd.concat([base, aug], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("base", len(base), "aug", len(aug), "out", len(out), "errors", len(all_errors))
    if all_errors:
        pd.DataFrame(all_errors, columns=["region", "T", "P_MPa", "error"]).to_csv(
            Path(args.out).with_suffix(".errors.csv"), index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":
    main()
