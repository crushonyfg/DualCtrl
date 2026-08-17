# 官方 Baseline Benchmark 报告

本报告只包含三个文献 baseline 与一个 oracle：

1. `kh_dual_control`：Klenske-Hennig approximate dual control；
2. `arcari_dual_smpc`：Arcari et al. dual stochastic MPC；
3. `tv_gp_lcb`：Bogunovic et al. TV-GP-UCB 的 cost-minimization LCB 版本；
4. `oracle_trend`：知道当前与未来 noiseless \(\theta\) trend 的 oracle planner。

所有非 oracle baseline 都只能使用同一个 nominal digital twin / simulator family 和同一批 physical observations。若实验设定为 `gap`，则 gap 只存在于 physical environment 中；baseline 不会被告知 true gap function。只有 oracle 使用 true physical dynamics / true gap。

## 实验配置说明

本次完整报告使用的是可在当前机器上跑完的较小配置：

- deployment horizon：30；
- seeds：2；
- scalar action grid size：9；
- CartPole action grid size：3；
- planning horizon：2；
- Arcari dual horizon：1；
- Arcari scenarios：1。

代码支持更大的 `--smpc-dual-horizon` 和 `--smpc-scenarios`，但 CartPole 上 explicit scenario tree + process-noise branching 计算量会快速增加。

注意：这里的 `oracle_trend` 是 numerical oracle / trend oracle，不是 continuous-action exact optimum。因此表中少数 `mean_oracle_regret < 0` 是数值近似和 coarse grid 造成的，不应解释为 baseline 真正优于理论 oracle。

---

## Scalar 主实验矩阵

| 环境 | Twin Gap | Regime | Baseline | 平均总成本 | 标准误 | 相对 Oracle Regret | n |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| scalar | gap | drifting | arcari_dual_smpc | 7.935 | 0.195 | -0.3283 | 2 |
| scalar | gap | drifting | kh_dual_control | 23.8 | 1.797 | 15.54 | 2 |
| scalar | gap | drifting | oracle_trend | 8.264 | 0.5233 | 0 | 2 |
| scalar | gap | drifting | tv_gp_lcb | 1.009e+04 | 2175 | 1.008e+04 | 2 |
| scalar | gap | piecewise | arcari_dual_smpc | 6.91 | 1.969 | -0.09042 | 2 |
| scalar | gap | piecewise | kh_dual_control | 39.95 | 16.71 | 32.95 | 2 |
| scalar | gap | piecewise | oracle_trend | 7.001 | 1.085 | 0 | 2 |
| scalar | gap | piecewise | tv_gp_lcb | 1.444e+04 | 1161 | 1.443e+04 | 2 |
| scalar | gap | static | arcari_dual_smpc | 4.796 | 0.4648 | 0 | 2 |
| scalar | gap | static | kh_dual_control | 39.95 | 16.71 | 35.15 | 2 |
| scalar | gap | static | oracle_trend | 4.796 | 0.4648 | 0 | 2 |
| scalar | gap | static | tv_gp_lcb | 1.303e+04 | 6143 | 1.303e+04 | 2 |
| scalar | no_gap | drifting | arcari_dual_smpc | 10.69 | 1.471 | 0.4107 | 2 |
| scalar | no_gap | drifting | kh_dual_control | 25.22 | 0.876 | 14.94 | 2 |
| scalar | no_gap | drifting | oracle_trend | 10.27 | 1.395 | 0 | 2 |
| scalar | no_gap | drifting | tv_gp_lcb | 8566 | 2165 | 8556 | 2 |
| scalar | no_gap | piecewise | arcari_dual_smpc | 6.788 | 2.185 | 1.067 | 2 |
| scalar | no_gap | piecewise | kh_dual_control | 39.95 | 16.71 | 34.23 | 2 |
| scalar | no_gap | piecewise | oracle_trend | 5.722 | 1.118 | 0 | 2 |
| scalar | no_gap | piecewise | tv_gp_lcb | 1542 | 241.3 | 1536 | 2 |
| scalar | no_gap | static | arcari_dual_smpc | 4.913 | 0.7838 | 0 | 2 |
| scalar | no_gap | static | kh_dual_control | 39.95 | 16.71 | 35.04 | 2 |
| scalar | no_gap | static | oracle_trend | 4.913 | 0.7838 | 0 | 2 |
| scalar | no_gap | static | tv_gp_lcb | 2774 | 704.2 | 2769 | 2 |

### Scalar 初步观察

- `arcari_dual_smpc` 在当前 coarse 配置下通常最接近 oracle；部分负 regret 来自 numerical oracle 近似，不代表真正超过 oracle。
- `kh_dual_control` 在当前 scalar deployment 设置下成本较高，说明该 paper 方法在这个连续部署设定中可能较保守或与当前 scalar formulation 不完全匹配。
- `tv_gp_lcb` 在控制任务上表现很差。这是严格 LCB acquisition 的结果，没有加入额外 stabilization heuristic。

---

## CartPole 主实验矩阵

| 环境 | Twin Gap | Regime | Baseline | 平均总成本 | 标准误 | 相对 Oracle Regret | 平均 Failure 数 | n |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cartpole | gap | drifting | arcari_dual_smpc | 0.9332 | 0.01372 | -0.1095 | 0 | 2 |
| cartpole | gap | drifting | kh_dual_control | 0.9332 | 0.01372 | -0.1095 | 0 | 2 |
| cartpole | gap | drifting | oracle_trend | 1.043 | 0.01605 | 0 | 0 | 2 |
| cartpole | gap | drifting | tv_gp_lcb | 839.4 | 0.4213 | 838.4 | 8 | 2 |
| cartpole | gap | piecewise | arcari_dual_smpc | 0.8535 | 0.02494 | -0.07415 | 0 | 2 |
| cartpole | gap | piecewise | kh_dual_control | 0.8535 | 0.02494 | -0.07415 | 0 | 2 |
| cartpole | gap | piecewise | oracle_trend | 0.9277 | 0.07752 | 0 | 0 | 2 |
| cartpole | gap | piecewise | tv_gp_lcb | 536.6 | 101 | 535.7 | 5 | 2 |
| cartpole | gap | static | arcari_dual_smpc | 0.8733 | 0.03486 | -0.05921 | 0 | 2 |
| cartpole | gap | static | kh_dual_control | 0.8733 | 0.03486 | -0.05921 | 0 | 2 |
| cartpole | gap | static | oracle_trend | 0.9325 | 0.02771 | 0 | 0 | 2 |
| cartpole | gap | static | tv_gp_lcb | 638.3 | 104.1 | 637.4 | 6 | 2 |
| cartpole | no_gap | drifting | arcari_dual_smpc | 0.6544 | 0.01547 | 0.1136 | 0 | 2 |
| cartpole | no_gap | drifting | kh_dual_control | 0.6544 | 0.01547 | 0.1136 | 0 | 2 |
| cartpole | no_gap | drifting | oracle_trend | 0.5408 | 0.005583 | 0 | 0 | 2 |
| cartpole | no_gap | drifting | tv_gp_lcb | 791.7 | 48.45 | 791.1 | 7.5 | 2 |
| cartpole | no_gap | piecewise | arcari_dual_smpc | 0.6232 | 0.03784 | 0.07591 | 0 | 2 |
| cartpole | no_gap | piecewise | kh_dual_control | 0.6232 | 0.03784 | 0.07591 | 0 | 2 |
| cartpole | no_gap | piecewise | oracle_trend | 0.5473 | 0.03465 | 0 | 0 | 2 |
| cartpole | no_gap | piecewise | tv_gp_lcb | 689.7 | 51.18 | 689.1 | 6.5 | 2 |
| cartpole | no_gap | static | arcari_dual_smpc | 0.6208 | 0.03263 | 0.1114 | 0 | 2 |
| cartpole | no_gap | static | kh_dual_control | 0.6208 | 0.03263 | 0.1114 | 0 | 2 |
| cartpole | no_gap | static | oracle_trend | 0.5095 | 0.05041 | 0 | 0 | 2 |
| cartpole | no_gap | static | tv_gp_lcb | 793 | 50.6 | 792.5 | 7.5 | 2 |

### CartPole 初步观察

- `arcari_dual_smpc` 和 `kh_dual_control` 在当前 coarse CartPole 配置下行动基本一致，因此结果相同。
- `tv_gp_lcb` 在 CartPole 上频繁 failure，导致成本极高。
- 当前 CartPole oracle 使用 coarse action grid，因此不是 exact continuous optimum；负 regret 仍然是 numerical approximation artifact。

---

## KH Section 6 reproduction

KH scalar reproduction curve 写在：

```text
kh_section6_curve.csv
```

当前 reproduction harness 输出：

```text
CE min u0=-0.6000, cost=1.7001
KH-dual min u0=-0.1500, cost=2.7665
```

这说明 KH dual control 的 root action 与 certainty-equivalent action 明显不同，符合 dual control toy example 中“CE 更激进、dual control 更考虑信息价值/不确定性”的定性现象。

---

## Stress panels

### Sparse physical data

文件：

```text
stress/sparse_physical_panel.csv
```

扫：

```text
m ∈ {1, 5, 10, 20}
```

其中 `m` 表示 physical observation / calibration update interval。所有 physical transitions 仍然计入真实累计成本，但只有每 `m` 步用于 posterior / GP update。

### Non-differentiable switch cost

文件：

```text
stress/nondiff_switch_panel.csv
```

扫：

```text
k ∈ {0, 0.05, 0.1, 0.2}
```

cost 使用：

\[
k \mathbf 1\{|a_t-a_{t-1}| > 0.05\}
\]

### Multimodal diagnostic notes

文件：

```text
stress/multimodal_notes.md
```

本轮没有随意实现新的 mixture controller。原因是：

- KH 和 TV-GP-LCB 的原生 belief assumptions 不直接等价于 mixture-regime controller；
- Arcari 可以处理 structural/model modes，但只有当同样的 nominal model set 对所有方法公平可用时才应启用；
- 不能把 true physical gap mode 只额外给 Arcari。

因此 multimodal 当前作为 assumption-stress-test 说明，而不是混入主表。

---

## 当前限制

1. **CartPole KH 仍是 actuator-gain belief 版本，不是完整 GP dynamics KH 版本。** 这仍然是后续需要补齐的最大方法实现点。
2. **Arcari full tree 支持更大 `L` 和 scenarios，但本次最终报告为保证可跑完使用了小配置。**
3. **TV-GP-LCB 是严格 bandit acquisition baseline；它在 control setting 中不稳定是结果本身，不应通过 heuristic 修改。**
4. **所有结果 seed 数较少，只用于 first-pass benchmark sanity，不应用作论文最终统计结果。**
