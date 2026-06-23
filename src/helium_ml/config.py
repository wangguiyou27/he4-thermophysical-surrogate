from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    data_path = Path(cfg["data_path"])
    if not data_path.is_absolute():
        data_path = (config_path.parent / data_path).resolve()
    cfg["data_path"] = str(data_path)
    return cfg

