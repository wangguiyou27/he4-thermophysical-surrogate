from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


MODEL_T_MIN = 2.2
MODEL_T_MAX = 300.0
MODEL_P_MIN = 0.001
MODEL_P_MAX = 3.0


def finite_stats(arr):
    x = np.asarray(arr, dtype=float).ravel()
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if x.size == 0:
        return None
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "n": int(x.size),
    }


def domain_fraction(temp, pressure_pa):
    t = np.asarray(temp, dtype=float).ravel()
    p_mpa = np.asarray(pressure_pa, dtype=float).ravel() / 1e6
    n = min(t.size, p_mpa.size)
    if n == 0:
        return 0, 0.0
    t = t[:n]
    p_mpa = p_mpa[:n]
    ok = (
        np.isfinite(t)
        & np.isfinite(p_mpa)
        & (t != 0)
        & (p_mpa != 0)
        & (t >= MODEL_T_MIN)
        & (t <= MODEL_T_MAX)
        & (p_mpa >= MODEL_P_MIN)
        & (p_mpa <= MODEL_P_MAX)
    )
    valid = np.isfinite(t) & np.isfinite(p_mpa) & (t != 0) & (p_mpa != 0)
    denom = int(valid.sum())
    return int(ok.sum()), float(ok.sum() / denom * 100.0) if denom else 0.0


def script_summary(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    keys = {}
    for name in ["T_R", "P_R", "TH", "T_INITIAL", "P_INITIAL", "F", "T_FINAL"]:
        m = re.search(rf"^\s*{name}\s*=\s*([^;\n]+)", text, flags=re.MULTILINE)
        if m:
            keys[name] = m.group(1).strip()
    calls = sorted(set(re.findall(r"refpropm\('([^']+)'", text)))
    return {"script": path.name, **keys, "refpropm_codes": ",".join(calls)}


def simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows found._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, sep, *rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scripts = pd.DataFrame(script_summary(p) for p in sorted(case_dir.glob("MicroPTC*.m")))
    scripts.to_csv(out / "microptc_script_summary.csv", index=False, encoding="utf-8-sig")

    rows = []
    for mat_path in sorted(case_dir.glob("Micro_pulse_tube*.mat")):
        d = loadmat(
            mat_path,
            variable_names=[
                "T_GAS",
                "PRESSURE",
                "T_GAS_FACE",
                "PRESSURE_FACE",
                "MONITOR_TEMPERATURE_ALL",
                "MONITOR_PRESSURE_ALL",
                "COOLING_POWER",
                "ACOUSTIC_POWER",
                "T_R",
                "P_R",
            ],
            squeeze_me=True,
        )
        t_stats = finite_stats(d.get("T_GAS", []))
        p_stats = finite_stats(np.asarray(d.get("PRESSURE", []), dtype=float) / 1e6)
        mt_stats = finite_stats(d.get("MONITOR_TEMPERATURE_ALL", []))
        mp_stats = finite_stats(np.asarray(d.get("MONITOR_PRESSURE_ALL", []), dtype=float) / 1e6)
        in_n, in_pct = domain_fraction(d.get("T_GAS", []), d.get("PRESSURE", []))
        face_n, face_pct = domain_fraction(d.get("T_GAS_FACE", []), d.get("PRESSURE_FACE", []))
        monitor_n, monitor_pct = domain_fraction(d.get("MONITOR_TEMPERATURE_ALL", []), d.get("MONITOR_PRESSURE_ALL", []))
        rows.append(
            {
                "case": mat_path.name,
                "T_GAS_min_K": t_stats["min"] if t_stats else np.nan,
                "T_GAS_max_K": t_stats["max"] if t_stats else np.nan,
                "PRESSURE_min_MPa": p_stats["min"] if p_stats else np.nan,
                "PRESSURE_max_MPa": p_stats["max"] if p_stats else np.nan,
                "MONITOR_T_min_K": mt_stats["min"] if mt_stats else np.nan,
                "MONITOR_T_max_K": mt_stats["max"] if mt_stats else np.nan,
                "MONITOR_P_min_MPa": mp_stats["min"] if mp_stats else np.nan,
                "MONITOR_P_max_MPa": mp_stats["max"] if mp_stats else np.nan,
                "state_domain_count": in_n,
                "state_domain_fraction_pct": in_pct,
                "face_domain_count": face_n,
                "face_domain_fraction_pct": face_pct,
                "monitor_domain_count": monitor_n,
                "monitor_domain_fraction_pct": monitor_pct,
                "cooling_power_W": float(np.asarray(d.get("COOLING_POWER", np.nan)).squeeze()),
                "acoustic_power_W": float(np.asarray(d.get("ACOUSTIC_POWER", np.nan)).squeeze()),
            }
        )
    mat_df = pd.DataFrame(rows)
    mat_df.to_csv(out / "microptc_mat_state_ranges.csv", index=False, encoding="utf-8-sig")

    required = pd.DataFrame(
        [
            {"refpropm_code": "L", "meaning": "thermal conductivity", "current_model": "covered"},
            {"refpropm_code": "C", "meaning": "isobaric heat capacity Cp", "current_model": "covered"},
            {"refpropm_code": "V", "meaning": "dynamic viscosity", "current_model": "covered"},
            {"refpropm_code": "D", "meaning": "density", "current_model": "covered"},
            {"refpropm_code": "^", "meaning": "Prandtl number", "current_model": "not covered"},
            {"refpropm_code": "Z", "meaning": "compressibility factor", "current_model": "not covered"},
            {"refpropm_code": "B", "meaning": "thermal expansion-related REFPROP output used in legacy code", "current_model": "not covered"},
        ]
    )
    required.to_csv(out / "microptc_required_properties.csv", index=False, encoding="utf-8-sig")

    md = [
        "# MicroPTC suitability report",
        "",
        f"Current helium surrogate domain: T={MODEL_T_MIN}-{MODEL_T_MAX} K, P={MODEL_P_MIN}-{MODEL_P_MAX} MPa.",
        "",
        "## Verdict",
        "",
        "The MicroPTC case is not directly suitable for the current surrogate model comparison.",
        "The saved simulations operate near 10 MPa, while the current surrogate is limited to 3 MPa.",
        "The legacy code also requires Prandtl number, compressibility factor, and an additional REFPROP output `B`, which are not current ANN targets.",
        "",
        "## State range summary",
        "",
        simple_markdown_table(mat_df),
        "",
        "## Required gas-property outputs",
        "",
        simple_markdown_table(required),
        "",
        "## Recommended use",
        "",
        "Use this case after extending the surrogate data/model to at least 12 MPa, preferably 14 MPa to match the coarser MicroPTC tables, and adding Pr, Z, and B or replacing the equations that need them with quantities available from the surrogate.",
    ]
    (out / "microptc_suitability_report.md").write_text("\n".join(md), encoding="utf-8")
    print(mat_df.to_string(index=False))
    print(f"Saved report to {out.resolve()}")


if __name__ == "__main__":
    main()
