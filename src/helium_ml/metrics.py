from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def relative_error(y_true: np.ndarray, y_pred: np.ndarray, y_floor: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(y_true), y_floor.reshape(1, -1))
    return np.abs(y_true - y_pred) / denom


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_floor: np.ndarray,
    target_cols: list[str],
) -> Dict:
    err = relative_error(y_true, y_pred, y_floor) * 100.0
    per_target = {}
    r2_values = []
    nrmse_values = []
    for i, col in enumerate(target_cols):
        e = err[:, i]
        residual = y_pred[:, i] - y_true[:, i]
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        y_span = float(np.max(y_true[:, i]) - np.min(y_true[:, i]))
        nrmse_range = float(rmse / y_span) if y_span > 0 else float("nan")
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        if np.isfinite(r2):
            r2_values.append(r2)
        if np.isfinite(nrmse_range):
            nrmse_values.append(nrmse_range)
        per_target[col] = {
            "mape": float(np.mean(e)),
            "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)),
            "p99": float(np.percentile(e, 99)),
            "max": float(np.max(e)),
            "mae": mae,
            "rmse": rmse,
            "nrmse_range": nrmse_range,
            "r2": r2,
            "within_1pct": float(np.mean(e < 1.0) * 100.0),
            "within_5pct": float(np.mean(e < 5.0) * 100.0),
        }

    all_e = err.reshape(-1)
    return {
        "overall": {
            "mape": float(np.mean(all_e)),
            "median": float(np.median(all_e)),
            "p95": float(np.percentile(all_e, 95)),
            "p99": float(np.percentile(all_e, 99)),
            "max": float(np.max(all_e)),
            "mean_r2": float(np.mean(r2_values)) if r2_values else float("nan"),
            "mean_nrmse_range": float(np.mean(nrmse_values)) if nrmse_values else float("nan"),
            "within_1pct": float(np.mean(all_e < 1.0) * 100.0),
            "within_5pct": float(np.mean(all_e < 5.0) * 100.0),
        },
        "per_target": per_target,
    }


def worst_point_table(
    X_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_floor: np.ndarray,
    input_cols: list[str],
    target_cols: list[str],
    n: int = 200,
) -> pd.DataFrame:
    err = relative_error(y_true, y_pred, y_floor) * 100.0
    flat_order = np.argsort(err.reshape(-1))[::-1][:n]
    rows = []
    for flat_idx in flat_order:
        row_idx, target_idx = np.unravel_index(flat_idx, err.shape)
        row = {col: float(X_raw[row_idx, i]) for i, col in enumerate(input_cols)}
        target = target_cols[target_idx]
        row.update(
            {
                "target": target,
                "true": float(y_true[row_idx, target_idx]),
                "pred": float(y_pred[row_idx, target_idx]),
                "abs_error": float(abs(y_pred[row_idx, target_idx] - y_true[row_idx, target_idx])),
                "relative_error_pct": float(err[row_idx, target_idx]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def region_metrics(
    temperatures: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_floor: np.ndarray,
    regions: Dict[str, list[float]],
) -> pd.DataFrame:
    err = relative_error(y_true, y_pred, y_floor) * 100.0
    rows = []
    for name, (lo, hi) in regions.items():
        mask = (temperatures >= lo) & (temperatures < hi)
        if not np.any(mask):
            continue
        e = err[mask].reshape(-1)
        rows.append(
            {
                "region": name,
                "t_min": lo,
                "t_max": hi,
                "n": int(mask.sum()),
                "mape": float(np.mean(e)),
                "p95": float(np.percentile(e, 95)),
                "p99": float(np.percentile(e, 99)),
                "max": float(np.max(e)),
                "within_5pct": float(np.mean(e < 5.0) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def physics_checks(frame: pd.DataFrame) -> Dict[str, float]:
    checks: Dict[str, float] = {}
    cp_cols = [c for c in frame.columns if c.lower() == "cp" or c.lower().startswith("cp ")]
    cv_cols = [c for c in frame.columns if c.lower() == "cv" or c.lower().startswith("cv ")]
    if cp_cols and cv_cols:
        cp = pd.to_numeric(frame[cp_cols[0]], errors="coerce")
        cv = pd.to_numeric(frame[cv_cols[0]], errors="coerce")
        valid = cp.notna() & cv.notna()
        if valid.any():
            checks["cp_ge_cv_fraction"] = float((cp[valid] >= cv[valid]).mean())
    density_cols = [c for c in frame.columns if c.lower().startswith("density")]
    if density_cols:
        rho = pd.to_numeric(frame[density_cols[0]], errors="coerce")
        checks["positive_density_fraction"] = float((rho > 0).mean())
    return checks
