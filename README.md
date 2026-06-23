# Reliability-aware neural surrogate for helium-4 thermophysical properties

This repository contains code, selected results, and publication/poster figures for a reliability-aware machine-learning surrogate of single-phase helium-4 thermophysical properties.

The project uses NIST REFPROP as the high-fidelity reference. The goal is not to replace REFPROP as a thermodynamic standard, but to study tail-error control, high-error point suppression, and reliability of neural thermophysical-property surrogates in cryogenic helium applications.

## Scope

- Fluid: helium-4
- Reference data source: NIST REFPROP
- State domain: 2.2-300 K, 0.001-3 MPa
- Phase domain: single-phase states only
- Inputs: temperature, pressure, phase code, reduced variables, and critical-feature descriptors
- Outputs:
  - density
  - entropy
  - enthalpy
  - speed of sound
  - Cv
  - Cp
  - viscosity
  - thermal conductivity

## Main result

The final local hybrid model achieved:

- MAPE: 0.1307%
- P99 relative error: 0.9414%
- Maximum relative error: 4.6514%
- Within 5% error: 100%
- Speedup over direct REFPROP calls: 17.45x for 100000 state points

The key reliability result is that local hybrid correction reduced the pure ANN maximum error from 46.10% to 4.65%, keeping all eight predicted properties below 5% maximum error on the final test split.

## Repository structure

```text
configs/        Experiment configuration files
src/            Reusable model and training utilities
scripts/        Data generation, training, evaluation, benchmark, and plotting scripts
runs/           Selected final metrics and benchmark outputs
paper_outputs/  Final tables, manuscript notes, poster figures, and SCI-style figures
data/           Data note and optional small samples only
```

## Important REFPROP note

REFPROP is not included in this repository. Users need a licensed REFPROP installation to regenerate the full dataset or rerun REFPROP benchmarks.

The full REFPROP-generated CSV datasets and trained model weights are intentionally excluded from this GitHub repository to avoid large files and licensing ambiguity. The included tables and figures provide the final reported results.

## Key outputs

Final publication/poster material:

```text
paper_outputs/he4_2p2_hybrid_paper/
```

Important subfolders:

```text
figures_sci/              SCI-style figures, PNG and SVG
poster_figures/           conference poster figures, PNG/SVG/PDF
poster_figures_tp_error/  T-P hybrid error heatmaps, PNG/SVG/PDF
tables/                   final result tables
```

Selected final metrics:

```text
runs/he4_extended_2p2_critical_features_grid_hybrid_v8/
runs/critical_feature_hybrid_refprop_benchmark_v8/
runs/baseline_comparison_full/
runs/precomputed_interpolation_benchmark/
```

## Example commands

Generate the dataset, if REFPROP is available:

```powershell
python scripts/generate_refprop_he4_extended.py --out data/he4data_single_phase_extended_2p2_300K.csv
python scripts/add_critical_features.py --input data/he4data_single_phase_extended_2p2_300K_margin5pct_phasecode.csv --out data/he4data_single_phase_extended_2p2_critical_features.csv
```

Train the critical-feature ANN:

```powershell
python scripts/train_supervised.py --config configs/he4_single_phase_extended_2p2_phaseaware_critical_features.json --out runs/he4_extended_2p2_phaseaware_critical_features_ann
```

Evaluate local hybrid correction:

```powershell
python scripts/evaluate_grid_hybrid.py --config configs/he4_single_phase_extended_2p2_phaseaware_critical_features.json --mlp-run runs/he4_extended_2p2_phaseaware_critical_features_ann --out runs/he4_extended_2p2_critical_features_grid_hybrid_v8 --save-all-errors
```

Regenerate poster figures:

```powershell
python scripts/make_poster_figures_from_handoff.py
python scripts/make_poster_tp_error_heatmap.py
```

## Citation and license

Please cite NIST REFPROP appropriately when using REFPROP-generated data or derived results. Add your preferred project license before public release if needed.
