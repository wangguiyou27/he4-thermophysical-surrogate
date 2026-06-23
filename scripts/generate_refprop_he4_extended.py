from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


def init_refprop(refprop_root: str):
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

    os.environ["RPPREFIX"] = refprop_root
    rp = REFPROPFunctionLibrary(refprop_root)
    rp.SETPATHdll(refprop_root)
    units = rp.GETENUMdll(0, "MOLAR BASE SI").iEnum
    molar_mass = rp.REFPROPdll("HELIUM", "", "M", units, 0, 0, 0, 0, [1.0]).Output[0]
    tcrit = rp.REFPROPdll("HELIUM", "", "TC", units, 0, 0, 0, 0, [1.0]).Output[0]
    pcrit = rp.REFPROPdll("HELIUM", "", "PC", units, 0, 0, 0, 0, [1.0]).Output[0] / 1e6
    return rp, units, molar_mass, float(tcrit), float(pcrit)


def stratified_tp(n: int, seed: int, t_min: float, t_max: float, p_min: float, p_max: float):
    rng = np.random.default_rng(seed)
    if t_min >= 5.0:
        bands = [
            (t_min, 20.0, 0.30),
            (20.0, 80.0, 0.25),
            (80.0, 200.0, 0.25),
            (200.0, t_max, 0.20),
        ]
    else:
        bands = [
            (t_min, 5.5, 0.35),
            (5.5, 20.0, 0.20),
            (20.0, 80.0, 0.20),
            (80.0, t_max, 0.25),
        ]
    temps = []
    pressures = []
    for i, (lo, hi, frac) in enumerate(bands):
        if hi <= lo:
            continue
        m = int(round(n * frac))
        sampler = qmc.LatinHypercube(d=2, seed=seed + i)
        u = sampler.random(m)
        temps.append(lo + (hi - lo) * u[:, 0])
        log_p = np.log10(p_min) + (np.log10(p_max) - np.log10(p_min)) * u[:, 1]
        pressures.append(10.0**log_p)
    t = np.concatenate(temps)
    p = np.concatenate(pressures)
    if len(t) > n:
        idx = rng.choice(np.arange(len(t)), size=n, replace=False)
        t = t[idx]
        p = p[idx]
    elif len(t) < n:
        extra_t = rng.uniform(t_min, t_max, size=n - len(t))
        extra_p = 10.0 ** rng.uniform(np.log10(p_min), np.log10(p_max), size=n - len(t))
        t = np.concatenate([t, extra_t])
        p = np.concatenate([p, extra_p])
    return t, p


def saturation_pressure_mpa(rp, units, temp_k: float):
    res = rp.REFPROPdll("HELIUM", "TQ", "P", units, 0, 0, float(temp_k), 0.0, [1.0])
    if res.ierr != 0:
        return np.nan
    return float(res.Output[0]) / 1e6


def classify_state(temp_k, pressure_mpa, psat_mpa, tcrit, pcrit, sat_margin):
    if temp_k < tcrit and np.isfinite(psat_mpa):
        rel = abs(pressure_mpa - psat_mpa) / max(psat_mpa, 1e-12)
        if rel < sat_margin:
            return "near_saturation"
        return "subcooled_liquid" if pressure_mpa > psat_mpa else "superheated_vapor"
    if temp_k >= tcrit and pressure_mpa >= pcrit:
        return "supercritical"
    return "single_phase_gas"


def refprop_props(rp, units, molar_mass, temp_k, pressure_mpa):
    prop_string = "D;S;H;W;CV;CP;ETA;TCX"
    res = rp.REFPROPdll("HELIUM", "TP", prop_string, units, 0, 0, float(temp_k), float(pressure_mpa) * 1e6, [1.0])
    if res.ierr != 0:
        return None, res.herr
    vals = res.Output
    out = {
        "Density (kg/m^3)": vals[0] * molar_mass,
        "Entropy (kJ/(kg·K))": vals[1] / molar_mass / 1e3,
        "Enthalpy (kJ/kg)": vals[2] / molar_mass / 1e3,
        "Speed of Sound (m/s)": vals[3],
        "Cv (kJ/(kg·K))": vals[4] / molar_mass / 1e3,
        "Cp (kJ/(kg·K))": vals[5] / molar_mass / 1e3,
        "Viscosity (μPa·s)": vals[6] * 1e6,
        "Thermal Conductivity (mW/(m·K))": vals[7] * 1e3,
    }
    if not all(np.isfinite(v) for v in out.values()):
        return None, "nonfinite output"
    return out, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../he4data_single_phase_extended_2p2_300K.csv")
    parser.add_argument("--summary", default="runs/he4_extended_generation_summary.json")
    parser.add_argument("--refprop-root", default=r"E:\REFPROP10\REFPROP")
    parser.add_argument("--n", type=int, default=360000)
    parser.add_argument("--t-min", type=float, default=2.2)
    parser.add_argument("--t-max", type=float, default=300.0)
    parser.add_argument("--p-min", type=float, default=0.001)
    parser.add_argument("--p-max", type=float, default=3.0)
    parser.add_argument("--sat-margin", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rp, units, molar_mass, tcrit, pcrit = init_refprop(args.refprop_root)
    temps, pressures = stratified_tp(args.n, args.seed, args.t_min, args.t_max, args.p_min, args.p_max)
    order = np.argsort(temps)
    temps = temps[order]
    pressures = pressures[order]

    rows = []
    errors = []
    state_counts: dict[str, int] = {}
    psat_cache: dict[int, float] = {}
    for i, (temp_k, pressure_mpa) in enumerate(zip(temps, pressures), start=1):
        if temp_k < tcrit:
            key = int(round(temp_k * 10000))
            psat = psat_cache.get(key)
            if psat is None:
                psat = saturation_pressure_mpa(rp, units, temp_k)
                psat_cache[key] = psat
        else:
            psat = np.nan
        state = classify_state(temp_k, pressure_mpa, psat, tcrit, pcrit, args.sat_margin)
        state_counts[state] = state_counts.get(state, 0) + 1
        if state == "near_saturation":
            continue
        props, err = refprop_props(rp, units, molar_mass, temp_k, pressure_mpa)
        if props is None:
            errors.append({"i": i, "T": float(temp_k), "P_MPa": float(pressure_mpa), "state": state, "error": str(err)})
            continue
        row = {
            "Temperature (K)": float(temp_k),
            "Pressure (MPa)": float(pressure_mpa),
            "phase_region": state,
            "Psat (MPa)": float(psat) if np.isfinite(psat) else np.nan,
        }
        row.update(props)
        rows.append(row)
        if i % 50000 == 0:
            print(f"processed={i}, accepted={len(rows)}, errors={len(errors)}")
            sys.stdout.flush()

    df = pd.DataFrame(rows)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    summary = {
        "out": str(out),
        "n_requested": args.n,
        "n_accepted": int(len(df)),
        "n_errors": int(len(errors)),
        "state_counts_before_error_filter": state_counts,
        "temperature_range": [float(df["Temperature (K)"].min()), float(df["Temperature (K)"].max())],
        "pressure_range": [float(df["Pressure (MPa)"].min()), float(df["Pressure (MPa)"].max())],
        "tcrit_K": tcrit,
        "pcrit_MPa": pcrit,
        "sat_margin": args.sat_margin,
        "elapsed_s": time.perf_counter() - t0,
        "errors_preview": errors[:20],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
