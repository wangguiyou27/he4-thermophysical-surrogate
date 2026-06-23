from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


@dataclass
class PreparedData:
    raw: pd.DataFrame
    X_raw: np.ndarray
    y_raw: np.ndarray
    X_scaled: np.ndarray
    y_scaled: np.ndarray
    y_proc: np.ndarray
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    y_floor: np.ndarray
    input_cols: list[str]
    target_cols: list[str]
    log_target_indices: list[int]
    shifted_log_target_offsets: dict[int, float]


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.weights is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.weights[idx]


def load_clean_frame(cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data_path"], skiprows=cfg.get("skip_rows", []), low_memory=False)
    aux_cols: list[str] = []
    for rule in cfg.get("conditional_sample_weights", []):
        aux_cols.extend(rule.get("bounds", {}).keys())
    cols = list(dict.fromkeys(cfg["input_cols"] + cfg["target_cols"] + aux_cols))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {cfg['data_path']}: {missing}")

    out = df[cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    for col in cfg["input_cols"]:
        out = out[out[col] > 0]
    return out.reset_index(drop=True)


def prepare_data(cfg: Dict) -> PreparedData:
    df = load_clean_frame(cfg)
    X_raw = df[cfg["input_cols"]].to_numpy(dtype=np.float64)
    y_raw = df[cfg["target_cols"]].to_numpy(dtype=np.float64)

    X_proc = X_raw.copy()
    for col in cfg.get("log_input_cols", []):
        idx = cfg["input_cols"].index(col)
        X_proc[:, idx] = np.log10(np.clip(X_proc[:, idx], 1e-300, None))

    y_proc = y_raw.copy()
    log_target_indices: list[int] = []
    shifted_log_target_offsets: dict[int, float] = {}
    for col in cfg.get("log_target_cols", []):
        idx = cfg["target_cols"].index(col)
        if np.any(y_proc[:, idx] <= 0):
            raise ValueError(f"Cannot log-transform non-positive target column: {col}")
        y_proc[:, idx] = np.log10(np.clip(y_proc[:, idx], 1e-300, None))
        log_target_indices.append(idx)
    for col, offset in cfg.get("shifted_log_target_offsets", {}).items():
        idx = cfg["target_cols"].index(col)
        offset = float(offset)
        shifted = y_proc[:, idx] + offset
        if np.any(shifted <= 0):
            raise ValueError(f"Offset for shifted log target is too small: {col} offset={offset}")
        y_proc[:, idx] = np.log10(np.clip(shifted, 1e-300, None))
        shifted_log_target_offsets[idx] = offset

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_proc)
    y_scaled = scaler_y.fit_transform(y_proc)

    q = float(cfg.get("relative_error_floor_quantile", 0.01))
    y_floor = np.quantile(np.abs(y_raw), q, axis=0)
    y_floor = np.maximum(y_floor, 1e-12)

    return PreparedData(
        raw=df,
        X_raw=X_raw,
        y_raw=y_raw,
        y_proc=y_proc,
        X_scaled=X_scaled,
        y_scaled=y_scaled,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        y_floor=y_floor,
        input_cols=list(cfg["input_cols"]),
        target_cols=list(cfg["target_cols"]),
        log_target_indices=log_target_indices,
        shifted_log_target_offsets=shifted_log_target_offsets,
    )


def split_indices(n: int, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train_val, test = train_test_split(
        idx,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("seed", 42),
        shuffle=True,
    )
    val_fraction = cfg.get("val_size", 0.1) / (1.0 - cfg.get("test_size", 0.2))
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=cfg.get("seed", 42),
        shuffle=True,
    )
    return train, val, test


def summarize_regions(temperatures: np.ndarray, regions: Dict[str, Iterable[float]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for name, bounds in regions.items():
        lo, hi = bounds
        summary[name] = int(((temperatures >= lo) & (temperatures < hi)).sum())
    return summary


def sample_weights_from_temperature(temperatures: np.ndarray, cfg: Dict) -> np.ndarray:
    regions = cfg.get("temperature_regions", {})
    region_weights = cfg.get("temperature_region_weights", {})
    weights = np.ones(len(temperatures), dtype=np.float64)
    for name, bounds in regions.items():
        lo, hi = bounds
        weights[(temperatures >= lo) & (temperatures < hi)] = float(region_weights.get(name, 1.0))
    raw = cfg.get("_raw_frame_for_weights")
    if raw is not None:
        for rule in cfg.get("conditional_sample_weights", []):
            mask = np.ones(len(raw), dtype=bool)
            for col, bounds in rule.get("bounds", {}).items():
                lo, hi = bounds
                vals = raw[col].to_numpy(dtype=np.float64)
                mask &= (vals >= float(lo)) & (vals <= float(hi))
            weights[mask] *= float(rule.get("weight", 1.0))
    weights = weights / np.mean(weights)
    return weights
