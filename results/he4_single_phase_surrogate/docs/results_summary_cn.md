# 2.2-300 K 单相氦-4 ANN/Hybrid 论文结果整理

## 1. 数据集与研究范围

最终论文建议采用 2.2-300 K、0.001-3 MPa 的单相氦-4数据作为主线。数据来自 REFPROP，并通过相区筛选去除两相区和近饱和风险点。

论文中建议使用：

- `tables/table_1_dataset_summary.csv`
- `tables/table_1_phase_counts.csv`
- `tables/table_1_variable_ranges.csv`
- `figures/fig_1_dataset_phase_map.png`

可写结论：

> The constructed dataset covers single-phase helium-4 states from 2.2 K to 300 K and from 0.001 MPa to 3 MPa, including subcooled liquid, superheated vapor, single-phase gas, and supercritical regions.

## 2. 最终模型整体精度

最终推荐模型是 critical-feature ANN + local hybrid correction。

核心结果来自：

- `tables/table_2_overall_ann_vs_hybrid.csv`

关键指标：

- Pure critical-feature ANN:
  - MAPE: 0.1655%
  - P99: 1.4251%
  - Max: 46.1014%
  - Within 5%: 99.8857%

- Local hybrid:
  - MAPE: 0.1307%
  - P99: 0.9414%
  - Max: 4.6514%
  - Mean R2: 0.9999845
  - Within 5%: 100%

可写结论：

> The local hybrid correction reduced the maximum error from 46.10% to 4.65%, while also improving the P99 error from 1.43% to 0.94%. This demonstrates that the proposed correction strategy mainly improves the tail reliability rather than merely reducing the average error.

## 3. 逐物性评价指标

逐物性指标来自：

- `tables/table_3_hybrid_per_property_metrics.csv`
- `figures/fig_3_per_property_error_metrics.png`
- `figures/fig_3_r2_and_within_thresholds.png`

所有物性的最大误差均小于 5%：

- Density: 4.1990%
- Entropy: 3.8192%
- Enthalpy: 4.6514%
- Speed of sound: 1.8482%
- Cv: 4.0036%
- Cp: 3.3807%
- Viscosity: 3.8245%
- Thermal conductivity: 1.9223%

可写结论：

> For all eight target properties, the maximum relative error is below 5%, indicating that the final model is not only accurate on average but also robust against isolated bad points.

## 4. Hybrid 修正效果

建议使用：

- `figures/fig_5_ann_vs_hybrid_max_error.png`
- `figures/fig_5_ann_vs_hybrid_overall_errors.png`
- `figures/fig_5_hybrid_worst_points_tp.png`
- `tables/table_3_ann_vs_hybrid_per_property.csv`
- `tables/table_3_hybrid_worst_points.csv`

写作重点：

- 纯 ANN 的平均误差已经很低，但在 Cp、Enthalpy、Density、Entropy 等局部区域仍有尾部坏点。
- Hybrid 不是为了替代 ANN，而是作为局部修正层，专门压制高风险区域的极端误差。
- 最终所有测试点误差均小于 5%。

可写结论：

> The hybrid correction is especially useful for engineering simulations, where a single large property error may destabilize an iterative calculation. By suppressing the tail errors, the hybrid model provides a more reliable alternative to the pure ANN.

## 5. Hybrid 全域误差热图

已重新导出全测试集 hybrid 误差，并生成热图：

- `figures/fig_4_hybrid_error_maps/fig_4a_hybrid_max_error_heatmap.png`
- `figures/fig_4_hybrid_error_maps/fig_4b_hybrid_mean_error_heatmap.png`
- `figures/fig_4_hybrid_error_maps/fig_4_density_error_heatmap.png`
- `figures/fig_4_hybrid_error_maps/fig_4_entropy_error_heatmap.png`
- `figures/fig_4_hybrid_error_maps/fig_4_enthalpy_error_heatmap.png`
- `figures/fig_4_hybrid_error_maps/fig_4_cp_error_heatmap.png`

对应全点误差数据：

- `tables/table_3_hybrid_all_point_errors.csv`

可写结论：

> The error maps show that the largest residual errors are mainly concentrated in low-temperature and near-critical regions, which are also the most nonlinear regions of helium thermophysical behavior.

## 6. 基线模型对比

基线结果来自：

- `tables/table_4_baseline_summary.csv`
- `figures/fig_6_baseline_accuracy_speed_tail.png`

已有模型：

- Current MLP
- Random Forest
- Extra Trees
- XGBoost
- Gaussian RBF Ridge

写作重点：

- Extra Trees、Random Forest、Gaussian RBF Ridge 的平均 MAPE 可以很低。
- 但这些模型的最大误差明显更大，且推理速度低于 MLP。
- 当前最终路线不是单纯追求最低平均 MAPE，而是追求速度、尾部可靠性和部署简洁性的综合最优。

可写结论：

> Although several tree-based and kernel-based models achieve lower mean errors, their larger maximum errors and lower inference speeds make them less attractive for iterative cryogenic engineering simulations.

## 7. 速度对比

速度结果来自：

- `tables/table_5_speed_summary.csv`
- `figures/fig_7_speed_benchmark.png`

100000 点结果：

- ANN throughput: 247985.7 points/s
- Hybrid throughput: 237308.6 points/s
- REFPROP throughput: 13600.4 points/s
- ANN speedup vs REFPROP: 18.23x
- Hybrid speedup vs REFPROP: 17.45x
- Hybrid slowdown vs ANN: 4.50%

可写结论：

> The local hybrid correction introduces only a 4.5% slowdown relative to pure ANN inference, while the hybrid model remains 17.45 times faster than direct REFPROP calls for 100000 state points.

## 8. Hybrid 修正区域说明

Hybrid 修正区域与路由点数来自：

- `tables/table_6_hybrid_grid_regions.csv`
- `tables/table_6_hybrid_routed_counts.csv`
- `figures/fig_2_hybrid_routed_counts.png`

写作重点：

- 局部修正区域主要围绕低温液体、近临界低压区、临界 Cp 峰、低密度气体和低温近零焓/熵区域。
- 这说明 hybrid 修正不是全域慢速替代，而是针对物性剧烈变化区域的局部补强。

## 9. 推荐正文图表顺序

建议正文使用：

1. Fig. 1：数据域和相区分布。
2. Fig. 2：模型结构和 hybrid 流程示意图。使用 `figures/fig_2_model_workflow.png` 或 `figures/fig_2_model_workflow.pdf`。
3. Fig. 3：逐物性误差和 R2/within-threshold 图。
4. Fig. 4：hybrid 全域误差热图。
5. Fig. 5：ANN 与 hybrid 尾部误差对比。
6. Fig. 6：机器学习基线模型对比。
7. Fig. 7：ANN/hybrid/REFPROP 速度对比。

建议正文使用：

1. Table 1：数据集范围和相区数量。
2. Table 2：ANN 与 hybrid 整体指标。
3. Table 3：hybrid 逐物性指标。
4. Table 4：基线模型对比。
5. Table 5：速度对比。

## 10. 目前还可以继续补强的材料

如果想让论文更完整，下一步建议做两件事：

1. 画一个更正式的模型流程图，替换或补充 `fig_2_hybrid_routed_counts.png`。
2. 写英文版 Method 和 Results 初稿，把这里的表图逐一嵌入。

不建议现在继续深入 GM 脉管算例。它会把论文重心从“高可靠物性代理模型”转移到“制冷机模型验证”，增加不必要的审稿风险。
