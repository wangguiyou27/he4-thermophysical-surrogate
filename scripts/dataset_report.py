from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helium_ml.config import load_config
from helium_ml.data import prepare_data, summarize_regions
from helium_ml.metrics import physics_checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)
    temp = data.raw[cfg["input_cols"][0]].to_numpy()
    report = {
        "name": cfg["name"],
        "data_path": cfg["data_path"],
        "n_rows": len(data.raw),
        "input_ranges": {
            c: [float(data.raw[c].min()), float(data.raw[c].max())] for c in cfg["input_cols"]
        },
        "target_ranges": {
            c: [float(data.raw[c].min()), float(data.raw[c].max())] for c in cfg["target_cols"]
        },
        "temperature_regions": summarize_regions(temp, cfg.get("temperature_regions", {})),
        "physics_checks": physics_checks(data.raw),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

