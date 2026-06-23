# He-4 2.2 K Hybrid Paper Assets

This folder contains manuscript-ready tables and figures for the 2.2-300 K single-phase helium-4 ANN/hybrid paper line.

## Core Result

- Hybrid overall MAPE: 0.1307%.
- Hybrid P99 error: 0.9414%.
- Hybrid maximum error: 4.6514%.
- Hybrid mean R2: 0.99998452.
- Within 5%: 100.0%.
- 100000-point hybrid speedup over REFPROP: 17.45x.
- Hybrid slowdown relative to ANN: 4.50%.

## Figure Mapping

- Fig. 1: `figures/fig_1_dataset_phase_map.png`
- Fig. 2: `figures/fig_2_model_workflow.png` or `figures/fig_2_model_workflow.pdf`
- Fig. 3: `figures/fig_3_per_property_error_metrics.png` and `figures/fig_3_r2_and_within_thresholds.png`
- Fig. 4: `figures/fig_4_hybrid_error_maps/`
- Fig. 5: `figures/fig_5_ann_vs_hybrid_max_error.png`, `figures/fig_5_ann_vs_hybrid_overall_errors.png`, `figures/fig_5_hybrid_worst_points_tp.png`
- Fig. 6: `figures/fig_6_baseline_accuracy_speed_tail.png`
- Fig. 7: `figures/fig_7_speed_benchmark.png`

## Table Mapping

- Table 1: `tables/table_1_dataset_summary.csv`, `tables/table_1_phase_counts.csv`, `tables/table_1_variable_ranges.csv`
- Table 2: `tables/table_2_overall_ann_vs_hybrid.csv`
- Table 3: `tables/table_3_hybrid_per_property_metrics.csv`
- Table 4: `tables/table_4_baseline_summary.csv`
- Table 5: `tables/table_5_speed_summary.csv`
- Additional hybrid details: `tables/table_6_hybrid_grid_regions.csv`, `tables/table_6_hybrid_routed_counts.csv`

## Note

The Fig. 4 heatmaps are based on the exported all-point hybrid errors from the final test split.
