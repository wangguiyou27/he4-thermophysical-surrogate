from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tcrit", type=float, default=5.1953)
    parser.add_argument("--pcrit", type=float, default=0.22832)
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    t = df["Temperature (K)"].to_numpy(dtype=np.float64)
    p = df["Pressure (MPa)"].to_numpy(dtype=np.float64)
    tau = (t - args.tcrit) / args.tcrit
    pi = (p - args.pcrit) / args.pcrit
    dist = np.sqrt(tau * tau + pi * pi)
    df["reduced_temperature"] = t / args.tcrit
    df["reduced_pressure"] = p / args.pcrit
    df["critical_distance"] = np.maximum(dist, 1e-6)
    df["inverse_critical_distance"] = np.minimum(1.0 / np.maximum(dist, 1e-4), 1e4)
    df["abs_reduced_temperature_offset"] = np.abs(tau) + 1e-6
    df["abs_reduced_pressure_offset"] = np.abs(pi) + 1e-6
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print({"rows": int(len(df)), "out": str(out)})


if __name__ == "__main__":
    main()
