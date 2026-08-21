# BRPC benchmark-validation horizon sweep 初步报告

本报告对应一次中等规模 validation sweep，而不是最终论文级结果。目标是检查：在 Toy2 使用 grid-DP 排除 CEM 搜索误差后，扩大 planning horizon $M$ 是否会改变 CE/PS baselines 的核心行为。

## 配置

```yaml
T: 60
planning_horizon_sweep: [1, 2, 3, 5, 10]
seeds: [0, 1, 2]
cold_start_transitions: 20
particles: 64
inducing_points: 24
Toy2 optimizer: grid_dp
Toy2 action_grid_size: 101
BOCPD hazard: 0.02
BOCPD max_experts: 8
BOCPD min_segment_length: 3
Toy2 changepoint: randomized Uniform(0.35T, 0.65T)
```

输出文件：

```text
horizon_sweep_summary.csv
M*/brpc_validation_raw.csv
M*/brpc_validation_seed_summary.csv
M*/brpc_validation_summary.csv
M*/toy2_changepoint_summary.csv
```

## Toy2：核心结论

Toy2 在所有 $M$ 下，`oracle_current` 和 `oracle_future` 都迁移到新 production region $a_R\approx0.8$。因此 Toy2 的长期收益并不是被 switching cost 阻止，也不是 grid-DP optimizer 找不到迁移路径。

但 CE baselines 在 $M=1,2,3,5,10$ 下全部保持在旧 production point：

| M | ce_brpc net | ce_bbrpc net | oracle_current net | CE gap vs oracle |
|---:|---:|---:|---:|---:|
| 1 | 73.764 | 73.764 | 82.194 | 8.430 |
| 2 | 73.764 | 73.764 | 82.194 | 8.430 |
| 3 | 73.764 | 73.764 | 82.194 | 8.430 |
| 5 | 73.764 | 73.764 | 82.194 | 8.430 |
| 10 | 73.764 | 73.764 | 82.194 | 8.430 |

Post-change action summary 更直接：

| M | baseline | post mean action | near old $a_L$ | near new $a_R$ | dist to new | post reward/step |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ce_brpc | 0.200 | 1.000 | 0.000 | 0.600 | 1.226 |
| 1 | ce_bbrpc | 0.200 | 1.000 | 0.000 | 0.600 | 1.226 |
| 1 | oracle_current | 0.800 | 0.000 | 1.000 | 0.000 | 1.528 |
| 10 | ce_brpc | 0.200 | 1.000 | 0.000 | 0.600 | 1.226 |
| 10 | ce_bbrpc | 0.200 | 1.000 | 0.000 | 0.600 | 1.226 |
| 10 | oracle_current | 0.800 | 0.000 | 1.000 | 0.000 | 1.528 |

这说明：**在当前 Toy2 设置下，CE 的 failure 不是 CEM optimization error，也不是 planning horizon 太短导致的；即使用 grid-DP 和 $M=10$，learned CE 仍 stuck at old production point。**

## Toy2：PS behavior

`ps_brpc` 与 CE 一样完全 stuck at old point；`ps_bbrpc` 随 horizon 增长有改善：

| M | ps_brpc net | ps_bbrpc net | ps_bbrpc near new | ps_bbrpc dist to new |
|---:|---:|---:|---:|---:|
| 1 | 73.764 | 73.764 | 0.000 | 0.600 |
| 2 | 73.764 | 73.472 | 0.060 | 0.564 |
| 3 | 73.764 | 76.954 | 0.429 | 0.343 |
| 5 | 73.764 | 76.583 | 0.345 | 0.393 |
| 10 | 73.764 | 78.425 | 0.690 | 0.186 |

这提示 BOCPD mixture + posterior sampling 在较长 horizon 下偶尔能采到支持迁移的 latent path，并通过 grid-DP 执行迁移；但它仍低于 oracle，且方差较大。当前 fixed-expert PS 还不是完整的 PS-change-path，因此下一步需要实现 changepoint-path sampling 版本。

## Toy1：当前 sweep 不能用于结论

Toy1 仍使用 CEM。结果显示 current oracle 随 $M$ 增长并不单调，甚至 $M=10$ 下 learned CE 有时超过 `oracle_current`：

| M | ce_brpc | ce_bbrpc | ps_brpc | ps_bbrpc | oracle_current | oracle_future |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -16.889 | -16.904 | -16.802 | -17.193 | -16.385 | -0.800 |
| 2 | -16.925 | -16.640 | -16.471 | -17.190 | -16.433 | -0.616 |
| 3 | -17.813 | -17.178 | -17.541 | -17.159 | -17.121 | -1.105 |
| 5 | -21.505 | -22.125 | -20.066 | -20.525 | -18.297 | -3.139 |
| 10 | -27.505 | -26.462 | -28.424 | -29.159 | -27.802 | -10.408 |

这更像 Toy1 CEM / objective / finite-horizon planning 没有收敛，而不是方法真实性能排序。Toy1 后续必须先做 optimizer convergence，并在小 $M$ 下用 dense grid / beam search 交叉验证 action。

## 当前判断

1. **Toy2 的 CE stuck 是结构性信号增强版**：排除 CEM 后，$M=10$ 仍 stuck；因此下一步应查 calibration / detectability / future belief adaptation，而不是继续调 CEM。
2. **Toy2 PS-BBRPC 有 horizon-dependent improvement**：说明 BOCPD mixture 里确实有部分 regime/change 信息能通过 sampling 转成迁移 action，但 fixed-expert PS 仍不够强，应实现 PS-change-path。
3. **Toy1 不能解释方法 ranking**：current oracle 本身没有稳定随 $M$ 改善，说明 Toy1 必须先做 CEM convergence / grid or beam cross-check。

## 下一步

1. 实现 `PS-change-path`，让 BOCPD-BRPC PS 在 horizon 内 sample future hazard/restart path；
2. 增加 `CE-MAP-expert`，检查 CE-BBRPC 是否被 mixture averaging 卡住；
3. 对 Toy2 做 forced-probe validation：强制经过 $a_D$ 后，BOCPD posterior 是否更新、CE 是否迁移；
4. 对 Toy1 做 optimizer convergence，而不是继续解释当前 horizon sweep ranking；
5. 然后再跑 Toy2 validation-scale：$T=200/240$、particles=128、inducing=32、grid=201、seeds=20。
