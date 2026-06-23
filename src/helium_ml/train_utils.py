from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ArrayDataset, PreparedData
from .model import MultiPropertyMLP


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Dict, data: PreparedData) -> MultiPropertyMLP:
    return MultiPropertyMLP(
        input_dim=len(data.input_cols),
        output_dim=len(data.target_cols),
        hidden_layers=list(cfg["hidden_layers"]),
        dropout=float(cfg.get("dropout", 0.05)),
    )


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict,
    sample_weights: np.ndarray | None = None,
) -> Tuple[nn.Module, list[dict]]:
    dev = device()
    model = model.to(dev)
    loader = DataLoader(
        ArrayDataset(X_train, y_train, sample_weights),
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=True,
        drop_last=False,
    )
    target_weights = np.ones(y_train.shape[1], dtype=np.float32)
    for col, weight in cfg.get("target_loss_weights", {}).items():
        if col in cfg["target_cols"]:
            target_weights[cfg["target_cols"].index(col)] = float(weight)
    target_weights = torch.as_tensor(target_weights / target_weights.mean(), dtype=torch.float32, device=dev)
    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.get("epochs", 300)))

    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=dev)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=dev)
    history = []
    best_state = None
    best_val = float("inf")
    patience = int(cfg.get("patience", 50))
    stale = 0

    for epoch in range(1, int(cfg.get("epochs", 300)) + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in loader:
            if len(batch) == 3:
                xb, yb, wb = batch
                wb = wb.to(dev).reshape(-1, 1)
            else:
                xb, yb = batch
                wb = None
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad()
            squared = (model(xb) - yb) ** 2
            weighted = squared * target_weights.reshape(1, -1)
            if wb is not None:
                weighted = weighted * wb
            loss = weighted.mean()
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(xb)
            seen += len(xb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv), yv).item())
        train_loss /= max(seen, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history


def predict_raw(model: nn.Module, X_scaled: np.ndarray, data: PreparedData, batch_size: int = 8192) -> np.ndarray:
    dev = device()
    model = model.to(dev)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):
            xb = torch.as_tensor(X_scaled[start : start + batch_size], dtype=torch.float32, device=dev)
            preds.append(model(xb).cpu().numpy())
    y_scaled = np.vstack(preds)
    y_proc = data.scaler_y.inverse_transform(y_scaled)
    y_raw = y_proc.copy()
    for idx in data.log_target_indices:
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx])
    for idx, offset in data.shifted_log_target_offsets.items():
        y_raw[:, idx] = np.power(10.0, y_raw[:, idx]) - offset
    return y_raw


def save_artifacts(out_dir: str | Path, model: nn.Module, data: PreparedData, cfg: Dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    joblib.dump(data.scaler_X, out / "scaler_X.joblib")
    joblib.dump(data.scaler_y, out / "scaler_y.joblib")
    joblib.dump(
        {
            "y_floor": data.y_floor,
            "input_cols": data.input_cols,
            "target_cols": data.target_cols,
            "log_target_indices": data.log_target_indices,
            "shifted_log_target_offsets": data.shifted_log_target_offsets,
            "config": cfg,
        },
        out / "metadata.joblib",
    )
