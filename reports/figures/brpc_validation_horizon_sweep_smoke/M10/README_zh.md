# BRPC trajectory / geometry diagnostic figures

这些图来自：

```text
reports/tables/brpc_validation_horizon_sweep_smoke/M10/brpc_validation_raw.csv
```

绘图命令：

```bash
PYTHONPATH=/mnt/bn/feed-quality-training/user/yxu/DualCtrl \
python -m experiments.plot_brpc_trajectories \
  --raw reports/tables/brpc_validation_horizon_sweep_smoke/M10/brpc_validation_raw.csv \
  --out-dir reports/figures/brpc_validation_horizon_sweep_smoke/M10 \
  --toy both \
  --action-grid-size 201 \
  --horizon-raw \
    reports/tables/brpc_validation_horizon_sweep_smoke/M1/brpc_validation_raw.csv \
    reports/tables/brpc_validation_horizon_sweep_smoke/M2/brpc_validation_raw.csv \
    reports/tables/brpc_validation_horizon_sweep_smoke/M3/brpc_validation_raw.csv \
    reports/tables/brpc_validation_horizon_sweep_smoke/M5/brpc_validation_raw.csv \
    reports/tables/brpc_validation_horizon_sweep_smoke/M10/brpc_validation_raw.csv
```

## 文件说明

- `toy2_representative_trajectory.png`：Toy2 representative seed 的 time-action heatmap。背景是真实 operating reward；叠加 stagewise greedy、包含 switching cost 的 full-horizon oracle、各方法 action，以及 changepoint 竖线和 BOCPD recent changepoint probability。
- `toy2_aggregate_trajectory.png`：Toy2 多 seed aggregate trajectory。背景为 mean-theta reward landscape；方法轨迹为 median，非 oracle 方法带 10%–90% action band。
- `toy2_belief_diagnostics.png`：Toy2 representative seed 的 belief 诊断，包括 true theta、theta posterior mean、BOCPD recent changepoint probability 和新 expert mass proxy。
- `toy2_horizon_small_multiples.png`：Toy2 在 $M=1,2,3,5,10$ 下的小多图，用同一类 reward geometry 背景直观看 horizon 对 action 迁移的影响。
- `toy1_representative_trajectory.png`：Toy1 representative seed 的 theta、state/reference、action、cumulative net reward 轨迹。

## 解读要点

1. Toy2 CE-BRPC / CE-BBRPC 在 grid-DP 和 $M=10$ 下仍长期停在旧 production region $a_L\approx0.2$。
2. Toy2 oracle trajectory 是包含 switching cost 的多步 DP oracle，而不是逐步 greedy optimum。
3. PS-BBRPC 在较长 horizon 下更容易迁移到新 region，但仍低于 oracle。
4. Toy1 图目前主要用于诊断轨迹形态；Toy1 的方法排序仍需先完成 CEM / optimizer convergence。
