from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from generate_refprop_he4_extended import init_refprop, refprop_props


def sample_box(n, t_bounds, p_bounds, seed, phase_code, phase_region, log_p=False):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n)
    t = t_bounds[0] + (t_bounds[1] - t_bounds[0]) * u[:, 0]
    if log_p:
        lp = np.log10(p_bounds[0]) + (np.log10(p_bounds[1]) - np.log10(p_bounds[0])) * u[:, 1]
        p = 10**lp
    else:
        p = p_bounds[0] + (p_bounds[1] - p_bounds[0]) * u[:, 1]
    return t, p, np.full(n, phase_code), phase_region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/he4_specialist_aug_2p2_v3.csv")
    parser.add_argument("--refprop-root", default=r"E:\REFPROP10\REFPROP")
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    rp, units, molar_mass, *_ = init_refprop(args.refprop_root)
    specs = [
        ("critical_cp_peak", 70000, (5.20, 5.36), (0.225, 0.265), 3, False),
        ("liquid_near_critical_cp", 60000, (5.05, 5.20), (0.220, 0.250), 4, False),
        ("vapor_near_critical_cp", 70000, (5.00, 5.23), (0.170, 0.230), 1, False),
        ("near_lambda_liquid_enthalpy", 60000, (2.20, 2.50), (0.85, 1.35), 4, False),
        ("cold_liquid_low_enthalpy", 50000, (2.20, 4.60), (0.08, 1.40), 4, False),
        ("entropy_zero_supercritical", 70000, (5.20, 6.25), (1.30, 3.00), 3, False),
        ("high_density_lowT_supercritical", 30000, (5.30, 6.20), (1.8, 3.0), 3, False),
        ("low_density_highT_gas", 30000, (180.0, 300.0), (0.001, 0.006), 2, True),
        ("near_critical_gas_density", 50000, (5.18, 5.36), (0.220, 0.260), 2, False),
        ("transport_highT_highP", 30000, (240.0, 300.0), (2.0, 3.0), 2, False),
    ]
    rows = []
    errors = []
    for i, (name, n, tb, pb, code, log_p) in enumerate(specs):
        temps, pressures, codes, phase_region = sample_box(n, tb, pb, args.seed + i, code, name, log_p)
        accepted = 0
        for t, p, c in zip(temps, pressures, codes):
            props, err = refprop_props(rp, units, molar_mass, t, p)
            if props is None:
                errors.append({"region": name, "T": float(t), "P": float(p), "error": str(err)})
                continue
            row = {
                "Temperature (K)": float(t),
                "Pressure (MPa)": float(p),
                "phase_region": phase_region,
                "Psat (MPa)": np.nan,
                "phase_code": int(c),
            }
            row.update(props)
            rows.append(row)
            accepted += 1
        print(name, "requested", n, "accepted", accepted)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(out.with_suffix(".errors.csv"), index=False, encoding="utf-8-sig")
    print("total", len(rows), "errors", len(errors), "out", out)


if __name__ == "__main__":
    main()
