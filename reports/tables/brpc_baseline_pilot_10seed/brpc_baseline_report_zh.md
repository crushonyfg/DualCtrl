# BRPC baseline 10-seed pilot 报告

本报告包含新的 BRPC baseline 矩阵：`ce_brpc`、`ps_brpc`、`ce_bbrpc`、`ps_bbrpc`，并加入 `oracle_current` 作为真实当前 dynamics 参考、`oracle_future` 作为 toy-only future-regime appendix ceiling。它写入独立目录，不合并旧的 KH / Arcari / TV-GP 结果。

## 实验范围

- 环境：Toy1 与 Toy2。
- 矩阵：CE/PS planner × BRPC/BOCPD-BRPC calibration。
- horizon=12，seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]，particles=32，inducing points=12。
- CEM：horizon=2，population=32，iterations=4。
- 这是 10-seed pilot，用于检查 benchmark 是否有区分度；仍不用于论文级统计结论。

## 数学建模摘要

Toy1 使用标量动态：digital twin 为 $f_{DT}(x,a;\theta)=\theta x+a$，physical transition 额外包含 $\beta\tanh(2x)$、动作非线性 discrepancy 与高斯噪声。净回报按真实部署轨迹记账：任务项 $-q_x(x-x^{ref})^2$ 减 energy cost $\lambda_E a^2$ 与 switching cost $\lambda_\Delta(a-a_{prev})^2$。

Toy2 使用连续动作 response landscape：digital twin 由左、右、诊断三个 Gaussian basis 组成，physical response 再加固定 discrepancy 与观测噪声。净回报为 response 减 energy 与 switching cost；状态按 benchmark 约定记录为上一步动作。

BRPC 使用固定 inducing/support set、参数粒子、particle-specific discrepancy mean 与 shared discrepancy covariance。参数权重通过 discrepancy-free likelihood 做 tempered update，discrepancy 使用 fixed-support GP 条件更新，并在 ESS 低时把参数粒子与 discrepancy mean 一起 resample。

BOCPD-BRPC 在 BRPC expert mixture 上做 prequential evidence、hazard restart 分支、expert mass 归一化与 pruning。CE planner 使用 posterior predictive mean；PS planner 每个 physical step 采样一条 coherent latent path 后规划。Toy2 planner 的 stage objective 使用同一个 physical accounting：predicted response 减 energy cost 与 previous-action switching cost。

## 结果摘要

| environment | baseline | n_seeds | mean_total_net_reward | mean_total_task_reward | mean_total_energy_cost | mean_total_switching_cost | mean_planner_queries_total | mean_restart_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Toy1 | ce_bbrpc | 10 | -2.839 | -2.73 | 0.01917 | 0.08998 | 3072 | 0.2 |
| Toy1 | ce_brpc | 10 | -2.878 | -2.763 | 0.01976 | 0.09517 | 3072 | 0 |
| Toy1 | oracle_current | 10 | -2.737 | -2.636 | 0.01808 | 0.08214 | 3072 | 0 |
| Toy1 | oracle_future | 10 | -0.4926 | -0.4205 | 0.01383 | 0.05823 | 3072 | 0 |
| Toy1 | ps_bbrpc | 10 | -3.153 | -3.013 | 0.02335 | 0.1175 | 3072 | 0.1 |
| Toy1 | ps_brpc | 10 | -2.844 | -2.71 | 0.02113 | 0.1128 | 3072 | 0 |
| Toy2 | ce_bbrpc | 10 | 14.35 | 14.38 | 0.02409 | 0.006351 | 3072 | 0 |
| Toy2 | ce_brpc | 10 | 14.29 | 14.32 | 0.02426 | 0.007211 | 3072 | 0 |
| Toy2 | oracle_current | 10 | 15.59 | 16.02 | 0.1758 | 0.2532 | 3072 | 0 |
| Toy2 | oracle_future | 10 | 15.62 | 16.03 | 0.1836 | 0.2269 | 3072 | 0 |
| Toy2 | ps_bbrpc | 10 | 13.39 | 13.73 | 0.05713 | 0.2758 | 3072 | 0.1 |
| Toy2 | ps_brpc | 10 | 13.98 | 14.09 | 0.03148 | 0.07589 | 3072 | 0 |

## Oracle gap 摘要

`oracle_current` 知道当前真实 dynamics/discrepancy，但不知道 future noise；它用于判断当前 baselines 相对“已知当前模型的 MPC”还有多少空间。`oracle_future` 只作为 toy-only appendix ceiling，不是可用 baseline。

| environment | baseline | mean total net reward | gap vs current oracle |
|---|---|---:|---:|
| Toy1 | ce_bbrpc | -2.839 | 0.102 |
| Toy1 | ce_brpc | -2.878 | 0.142 |
| Toy1 | ps_bbrpc | -3.153 | 0.417 |
| Toy1 | ps_brpc | -2.844 | 0.107 |
| Toy1 | oracle_current | -2.737 | 0.000 |
| Toy1 | oracle_future | -0.493 | -2.244 |
| Toy2 | ce_bbrpc | 14.349 | 1.245 |
| Toy2 | ce_brpc | 14.287 | 1.307 |
| Toy2 | ps_bbrpc | 13.392 | 2.202 |
| Toy2 | ps_brpc | 13.983 | 1.611 |
| Toy2 | oracle_current | 15.594 | 0.000 |
| Toy2 | oracle_future | 15.622 | -0.028 |

Toy1 中 future-regime oracle 明显高于 current oracle，说明如果未来 drift path 可知，会极大改变最优行为；它只能作为 ceiling。Toy2 中 current/future oracle 接近，因为 changepoint path 在当前 toy 设置下较快变得可利用。

## Toy2 changepoint 行为分析

Toy2 horizon=12，默认 changepoint 在 $t=6$。下面统计 changepoint 后 $t\ge6$ 的 action 行为：

| baseline | post mean action | near old $a_L$ | near diag $a_D$ | near new $a_R$ | mean distance to new | post mean net reward/step |
|---|---:|---:|---:|---:|---:|---:|
| ce_bbrpc | 0.200 | 1.000 | 0.000 | 0.000 | 0.600 | 1.195 |
| ce_brpc | 0.199 | 1.000 | 0.000 | 0.000 | 0.601 | 1.186 |
| ps_bbrpc | 0.200 | 1.000 | 0.000 | 0.000 | 0.600 | 1.188 |
| ps_brpc | 0.198 | 1.000 | 0.000 | 0.000 | 0.602 | 1.204 |
| oracle_current | 0.714 | 0.117 | 0.033 | 0.850 | 0.094 | 1.400 |
| oracle_future | 0.740 | 0.067 | 0.050 | 0.883 | 0.069 | 1.401 |

这说明当前 CE/PS × BRPC/BOCPD-BRPC baselines 在 Toy2 changepoint 后几乎全部 stuck 在旧 production point $a_L\approx0.2$，而 oracle 会迁移到新 production region $a_R\approx0.8$。这是一个有用的 diagnostic signal：Toy2 pilot 已经显示出 evolving/regime-change + switching-cost setting 下存在可优化空间。但这仍是 10-seed pilot，后续需要检查是否由 calibration update、planner horizon、hazard 参数或 reward geometry 导致。

## 给外部分析的重点问题：为什么 Toy2 baseline fail？

当前最明确的 failure mode 是：真实 regime 已在 $t=6$ 从 old 切到 new，但四个 learned baselines 的 post-change action 仍几乎全部留在 $a_L\approx0.2$。结合 geometry gate，这不是因为 $a_R$ 没有更高 operating reward；`oracle_current` 在同一 reward/cost accounting 下会迁移到 $a_R$。因此外部分析可以重点看以下几类可能原因：

1. **Calibration signal 没有进入生产决策区域**：Toy2 的 diagnostic action $a_D=0.5$ 信息最高，但 CE/PS baselines 很少主动走到 $a_D$；如果 post-change 数据几乎都来自 $a_L$，BRPC/BOCPD-BRPC 即使形式上更新，也可能无法获得足够的 regime-discriminating evidence。
2. **BOCPD restart 未转化成 action migration**：calibration-only abrupt-change validation 中 BOCPD-BRPC 能 restart；但 closed-loop pilot 中 `ce_bbrpc` / `ps_bbrpc` 仍 stuck at old action，说明问题可能不只是 changepoint detection，而是 restart 后 posterior/predictive mean 仍不足以让 planner 跨过 switching barrier。
3. **短 horizon + switching cost 造成局部最优**：从 $a_L$ 到 $a_R$ 需要付 switching cost，$a_D$ 虽有信息价值但 one-step production reward 不一定最高。CE/PS 都没有显式 information value，因此 finite-horizon CEM 可能偏向“不动”。这正是 dual-control benchmark 需要暴露的结构。
4. **Posterior sampling 方差没有变成有效探索**：PS 在 Toy2 中反而比 CE 差，且 switching cost 更高，说明当前 posterior samples 可能带来噪声式切换，而不是系统性 diagnostic probing。
5. **Reward geometry 仍需 ablation**：虽然 production optimum / one-step net reward 已经分开，但还需要 sweep $\lambda_\Delta$、diagnostic reward margin、hazard、planner horizon，确认 failure 不是某个参数过强导致的 trivial stuck。

建议 GPT/外部 reviewer 优先检查 raw CSV 中 changepoint 前后 action、calibration output、true theta、restart_count、predicted_reward 与 realized_reward 的关系，尤其是 `brpc_baseline_raw.csv` 和 `toy2_changepoint_action_summary.csv`。

对应额外 CSV：

```text
pilot_oracle_gap_summary.csv
toy2_changepoint_action_summary.csv
```

## 输出文件

- `brpc_baseline_raw.csv`：逐步 raw 记录，包含 reward/cost decomposition、真实 theta、动作、累计回报、planner query count 和 calibrator diagnostics。
- `brpc_baseline_seed_summary.csv`：每个 environment/seed/baseline 的累计分项。
- `brpc_baseline_summary.csv`：按 environment/baseline 聚合的 mean 与 standard error。
- `pilot_oracle_gap_summary.csv`：每个 baseline 相对 `oracle_current` 的 total-net-reward gap。
- `toy2_changepoint_action_summary.csv`：Toy2 changepoint 后 action stickiness / near-old / near-new 统计。
- `config.json`：本次 pilot 配置。

## Caveats

- 该运行仍是 10-seed pilot，不是论文级统计；没有 bootstrap CI 或正式 paired significance protocol。
- `oracle_current` 已接入，用于参考当前真实 dynamics 下的 MPC ceiling；`oracle_future` 是 toy-only appendix ceiling，不是可部署 baseline。
- Toy2 geometry/reporting 区分 production operating optimum（不含 switching）和 previous-action-dependent one-step net reward；二者不应混用。
- 当前 Toy2 baseline stuck at old production point 可能来自 calibration/restart 未充分响应、planner horizon 太短、CEM objective 局部最优或 reward geometry 太容易卡住，需要后续 ablation。
- 未做 hazard / pruning / restart-margin sensitivity；也尚未接入 CartPole BRPC suite。
