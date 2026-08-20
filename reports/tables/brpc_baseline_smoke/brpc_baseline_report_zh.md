# BRPC baseline 轻量实验报告

本报告对应 `benchmark_brpc_baseline_implementation_spec.md` 中第一阶段的 BRPC baseline suite。它只包含 calibration/planner baseline，不包含 proposed planner，也不混入旧的 KH / Arcari / TV-GP 结果。

当前报告是 **smoke/small run**，目标是确认：

1. Toy1 / Toy2 benchmark 接口可运行；
2. BRPC 与 BOCPD-BRPC 的 belief update / diagnostics 可记录；
3. CE 与 posterior sampling planner 能形成 spec 中的 $2\times2$ baseline matrix；
4. reward/cost decomposition、query count、true theta、restart diagnostics 等 CSV 记账完整。

它不是论文级统计结果，不能据此声称某个 baseline 已经显著优于另一个 baseline。

---

## 1. Baseline 矩阵

本报告只跑四个 baseline：

| Calibration | CE planner | Posterior-sampling planner |
|---|---:|---:|
| BRPC | `ce_brpc` | `ps_brpc` |
| BOCPD-BRPC | `ce_bbrpc` | `ps_bbrpc` |

其中：

- **CE planner** 使用当前 belief 的 posterior predictive mean 做 planning，不使用单一 plug-in particle；
- **PS planner** 每个 physical step 从 belief 中 sample 一条 coherent latent path，然后在 sampled latent path 下 planning；
- **BRPC** 只做 gradual drift tracking，不显式建 changepoint；
- **BOCPD-BRPC** 是 restart-time expert mixture，用于 abrupt change / segment restart；BOCPD-PS 先 sample expert，再在该 expert 内 sample BRPC latent path。这里没有模拟未来尚未观测到的 changepoint，因此是 fixed-expert Thompson MPC，而不是 changepoint lookahead planner。

第一阶段明确不包含：

- proposed dual planner；
- information bonus；
- UCB/LCB；
- KH / Arcari / scenario tree；
- model-free RL / meta-RL；
- CVaR / safety theory。

---

## 2. 数学建模摘要

### 2.1 统一 physical objective

所有方法评估真实 deployment trajectory 上的累计净回报：

$$
J_T
=
\sum_{t=0}^{T-1}
\left[
 r_{\mathrm{task}}(s_t,a_t,s_{t+1})
 - c_E(a_t)
 - c_{\mathrm{sw}}(a_{t-1},a_t)
\right]
+r_T^{\mathrm{term}}.
$$

其中：

$$
c_E(a_t)=\lambda_E\|a_t\|_2^2,
\qquad
c_{\mathrm{sw}}(a_{t-1},a_t)=\lambda_\Delta\|a_t-a_{t-1}\|_2^2.
$$

报告中的 `total_net_reward` 是上述真实 physical reward 的累计值。所有 simulator queries 只用于 planning，不直接计入 physical reward；但会记录 query count。

---

### 2.2 Toy1：Delayed-Excitation Dynamics

Toy1 是一维动态系统。Digital twin 为：

$$
f_{\mathrm{DT}}(x_t,a_t;\theta_t)=\theta_t x_t+a_t.
$$

Physical system 为：

$$
x_{t+1}
=
\theta_t x_t+a_t
+\beta_t\tanh(2x_t)
+\kappa_\delta a_t|a_t|
+w_t,
\qquad
w_t\sim\mathcal N(0,\sigma_w^2).
$$

关键 geometry 是：

$$
\frac{\partial f_{\mathrm{DT}}}{\partial \theta_t}=x_t.
$$

因此在 $x_t\approx0$ 的 exploitation 区域，参数信息很弱；需要先用 action 把系统推离低敏感区域，下一步 transition 才产生 calibration value。这用于测试：短视 CE / PS 是否会长期停留在低信息区域。

Reward 为：

$$
r_t
=
-q_x(x_t-x_t^{\mathrm{ref}})^2
-\lambda_Ea_t^2
-\lambda_\Delta(a_t-a_{t-1})^2.
$$

---

### 2.3 Toy2：Stealth Changepoint Operating Landscape

Toy2 是无显式动态状态的 response landscape，状态记作上一时刻 action：

$$
s_t=a_{t-1}.
$$

动作 $a_t\in[0,1]$。Digital twin response 为：

$$
f_{\mathrm{DT}}(a;\theta)
=
b_L\phi_{a_L}(a)
+(b_R+c_R\theta)\phi_{a_R}(a)
+(b_D+c_D\theta)\phi_{a_D}(a).
$$

Physical response：

$$
y_t=f_{\mathrm{DT}}(a_t;\theta_t)+\delta_t(a_t)+\epsilon_t.
$$

Reward 为：

$$
r_t=y_t-\lambda_Ea_t^2-\lambda_\Delta(a_t-a_{t-1})^2.
$$

Toy2 的设计目标是让：

- old regime 的 production optimum 在 $a_L$ 附近；
- new regime 的 production optimum 在 $a_R$ 附近；
- diagnostic action $a_D$ 参数敏感度最高，但它本身不是任何单一 $\theta$ 下的 production optimum；
- $a_D$ 位于 $a_L$ 和 $a_R$ 之间，因此有 switching bridge 作用。

---

## 3. BRPC 与 BOCPD-BRPC 实现摘要

### 3.1 BRPC belief

BRPC belief 结构为：

$$
B_t^{\mathrm{BRPC}}
=
\left\{
(\theta_t^{(i)},w_t^{(i)},m_t^{(i)})_{i=1}^N,
C_t,
Z,
\phi
\right\}.
$$

当前轻量实现包含：

- 参数粒子 $\theta_t^{(i)}$；
- log-domain particle weights；
- 固定 inducing/support set $Z$；
- particle-specific discrepancy inducing mean $m_t^{(i)}$；
- shared discrepancy covariance $C_t$；
- ESS-based coupled resampling，即 $\theta^{(i)}$ 与 $m^{(i)}$ 一起 resample；
- predictive mixture；
- discrepancy-free theta likelihood update；
- fixed-support GP discrepancy conditioning。

### 3.2 BOCPD-BRPC belief

BOCPD-BRPC 是 BRPC expert mixture：

$$
B_t^{\mathrm{CP}}
=
\left\{
(\alpha_{e,t},s_e,B_{e,t}^{\mathrm{BRPC}})
:e\in\mathcal E_t
\right\},
\qquad
\sum_e\alpha_{e,t}=1.
$$

实现包含：

- prequential evidence：先用 update 前 expert 计算 predictive evidence；
- fresh restart expert；
- hazard restart branch；
- log-domain expert mass normalization；
- pruning；
- hard-anchor restart rule；
- 每个 retained expert 每步 observation 只 assimilate 一次。

### 3.3 CE planner

CE planner 使用 posterior predictive mean：

$$
a_t^{\mathrm{CE}}
=
\arg\max_a \widehat J_M(a;\mathbb E_{B_t}[f(s,a;\theta)+\delta(s,a)]).
$$

这里的 mean 是 posterior predictive mean，不是选取一个 mean particle 后 plug-in rollout。当前实现用 CEM 近似优化 finite-horizon return。

### 3.4 Posterior sampling planner

PS planner 每个 physical step sample 一条 horizon 内 coherent latent path：

$$
(\theta_{t:t+M}^{\star},\delta_{t:t+M}^{\star})\sim B_t,
$$

然后在 sampled latent path 下做 CEM planning：

$$
a_t^{\mathrm{PS}}
=
\arg\max_a J_M(a;\theta_{t:t+M}^{\star},\delta_{t:t+M}^{\star}).
$$

这避免了早期版本中“freeze 当前 sampled model 不随 horizon 演化”的问题。BOCPD-BRPC 下先 sample expert $e\sim\alpha$，再从该 expert 的 BRPC belief 中 sample latent path；该实现不 rollout future changepoint，只是 fixed-expert Thompson MPC。

---

## 4. Geometry gate 输出

Geometry screening CSV 已生成：

```text
reports/tables/brpc_geometry/toy1_geometry_screening.csv
reports/tables/brpc_geometry/toy2_geometry_screening.csv
```

每个 grid point 记录：

- expected reward；
- parameter sensitivity；
- Fisher / variance-reduction proxy；
- old/new predictive KL；
- switching cost。

### 4.1 Toy1 geometry 观察

Toy1 的参数敏感度为 $|x|$。生成结果显示：

- 在 $x=0$ 时：
  - `parameter_sensitivity = 0`；
  - `predictive_kl_old_new = 0`；
- 在 $x=\pm1$ 时：
  - `parameter_sensitivity = 1`；
  - old/new predictive KL 明显增大。

这符合 Delayed-Excitation 设计目标：quiet/exploitation 区域信息弱，需要先移动到有激励的状态后，后续 transition 才有 calibration value。

### 4.2 Toy2 geometry 观察

Toy2 现在明确区分三类量：

1. **operating reward** $r_{op}(a;\theta)=\mathbb E[y|a,\theta]-\lambda_Ea^2$，用于定义 production optimum；
2. **one-step net reward** $r_{1}(a;\theta,a_{prev})=r_{op}(a;\theta)-\lambda_\Delta(a-a_{prev})^2$，用于分析从某个 previous action 出发的单步收益；
3. **finite-horizon value**，由 planner 使用相同 stage reward 通过 CEM 近似优化。

因此 production optimum 不再和 previous-action-dependent switching cost 混用。重新生成的 `toy2_geometry_screening.csv` 关键 grid 点如下：

| action | old operating | new operating | old one-step net | new one-step net | parameter sensitivity | old/new KL | switching cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $a_L=0.2$ | 1.227 | 1.228 | 1.195 | 1.196 | $1.76\times10^{-6}$ | 0.00079 | 0.032 |
| $a_D=0.5$ | -0.261 | 1.090 | -0.461 | 0.890 | 2.252 | 1013.57 | 0.200 |
| $a_R=0.8$ | 0.728 | 1.540 | 0.216 | 1.028 | 0.812 | 365.57 | 0.512 |

全 grid 上：

- old production optimum（operating reward argmax）在 $a\approx0.197$；
- new production optimum（operating reward argmax）在 $a\approx0.7975$；
- diagnostic action $a_D=0.5$ 的 parameter sensitivity 和 old/new KL 最大；
- $a_D$ 相对 production optimum 的 operating margin 为 old regime 约 1.489、new regime 约 0.450，因此它不是任何单一 sampled model 下的 production optimum。

这说明 Toy2 通过了核心 geometry gate：diagnostic action $a_D$ 信息最高，但不是 old/new production optimum；同时它在 $a_L$ 与 $a_R$ 之间，有 switching bridge 结构。

---

## 5. Calibration-only validation gate

Calibration-only validation 不调用 planner，而是对 BRPC 与 BOCPD-BRPC 喂入完全相同的 exogenous calibration action sequence，用于单独验证 estimator/restart behavior。输出文件：

```text
reports/tables/brpc_calibration_validation_smoke/calibration_validation.csv
```

当前 smoke 配置为 horizon=40、particles=32，覆盖 Toy1/Toy2 的 stationary、gradual drift、abrupt change、no-change false-alarm 四种场景。关键 sanity check：

| toy | scenario | algorithm | theta RMSE | response RMSE | restart count | false-alarm rate | restart recall |
|---|---|---|---:|---:|---:|---:|---:|
| Toy1 | abrupt_change | BRPC | 0.146 | 0.116 | 0 | 0.0 | — |
| Toy1 | abrupt_change | BOCPD-BRPC | 0.079 | 0.058 | 1 | 0.0 | 1.0 |
| Toy1 | no_change_false_alarm | BRPC | 0.088 | 0.015 | 0 | 0.0 | — |
| Toy1 | no_change_false_alarm | BOCPD-BRPC | 0.099 | 0.014 | 0 | 0.0 | — |
| Toy2 | abrupt_change | BRPC | 0.472 | 0.370 | 0 | 0.0 | — |
| Toy2 | abrupt_change | BOCPD-BRPC | 0.133 | 0.185 | 1 | 0.0 | 1.0 |
| Toy2 | no_change_false_alarm | BRPC | 0.079 | 0.118 | 0 | 0.0 | — |
| Toy2 | no_change_false_alarm | BOCPD-BRPC | 0.077 | 0.119 | 0 | 0.0 | — |

这说明在当前 toy smoke 配置下，BOCPD-BRPC 能在 abrupt change 中触发 restart，同时在 deterministic no-change false-alarm control 中没有触发误报。该表只用于 estimator gate，不用于比较 CE/PS planner。

---

## 6. Current-oracle CEM convergence gate

为了避免“oracle 自己没优化好”导致 baseline 结论失真，先对 Toy2 current-dynamics oracle 做 CEM population/iterations sweep。输出文件：

```text
reports/tables/brpc_cem_convergence_smoke/brpc_cem_convergence_raw.csv
reports/tables/brpc_cem_convergence_smoke/brpc_cem_convergence_seed_summary.csv
reports/tables/brpc_cem_convergence_smoke/brpc_cem_convergence_summary.csv
```

当前 smoke sweep：Toy2、`oracle_current`、horizon=6、seeds=[0,1,2]、population $\in\{16,32\}$、iterations $\in\{2,4\}$。按 seed 平均的 total realized reward：

| CEM population | CEM iterations | mean total realized reward | min | max | queries/seed |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 6.279 | 5.689 | 6.638 | 384 |
| 16 | 4 | 7.029 | 6.882 | 7.288 | 768 |
| 32 | 2 | 6.645 | 6.139 | 7.378 | 768 |
| 32 | 4 | 7.465 | 7.233 | 7.773 | 1536 |

趋势符合预期：更大的 CEM budget 提高 current oracle 的 realized reward。正式 baseline pilot 前应至少使用已通过 oracle convergence smoke 的 CEM budget；不能用过小 CEM budget 做强结论。

---

## 7. Smoke 实验配置

当前 BRPC baseline smoke run 配置：

- environments：Toy1, Toy2；
- baselines：`ce_brpc`, `ps_brpc`, `ce_bbrpc`, `ps_bbrpc`；
- horizon：3；
- seeds：`[0]`；
- cold-start transitions：2；
- particles：8；
- inducing points：6；
- CEM horizon：2；
- CEM population：4；
- CEM iterations：1。

该配置只用于检查接口与记账，不用于论文统计结论。

---

## 8. Smoke 结果摘要

| environment | baseline | calibration | planner | n_seeds | horizon | cold_start_transitions | mean_total_task_reward | mean_total_energy_cost | mean_total_switching_cost | mean_total_net_reward | mean_planner_queries_total | mean_restart_count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Toy1 | ce_bbrpc | BOCPD-BRPC | CE | 1 | 3 | 2 | -0.09097 | 0.06163 | 0.6916 | -0.8442 | 24 | 0 |
| Toy1 | ce_brpc | BRPC | CE | 1 | 3 | 2 | -0.1219 | 0.06506 | 0.6943 | -0.8812 | 24 | 0 |
| Toy1 | ps_bbrpc | BOCPD-BRPC | PS | 1 | 3 | 2 | -1.406 | 0.06698 | 0.5152 | -1.988 | 24 | 0 |
| Toy1 | ps_brpc | BRPC | PS | 1 | 3 | 2 | -0.2812 | 0.01000 | 0.1007 | -0.3919 | 24 | 0 |
| Toy2 | ce_bbrpc | BOCPD-BRPC | CE | 1 | 3 | 2 | 1.204 | 0.01567 | 0.02042 | 1.168 | 24 | 1 |
| Toy2 | ce_brpc | BRPC | CE | 1 | 3 | 2 | 1.381 | 0.002671 | 0.02983 | 1.348 | 24 | 0 |
| Toy2 | ps_bbrpc | BOCPD-BRPC | PS | 1 | 3 | 2 | 1.847 | 0.005593 | 0.09557 | 1.746 | 24 | 0 |
| Toy2 | ps_brpc | BRPC | PS | 1 | 3 | 2 | 1.483 | 0.03455 | 0.1057 | 1.343 | 24 | 0 |

### 8.1 Toy1 smoke 观察

- `ps_brpc` 在该单 seed、短 horizon 下 net reward 最高；
- `ps_bbrpc` 表现最差，主要来自 task reward 很低；
- BOCPD-BRPC 在 horizon=3 且无明显 changepoint 的场景下没有优势；
- 该结果不应解读为 PS-BRPC 已经优于 CE-BRPC，只说明四条路径可运行且记账完整。

### 8.2 Toy2 smoke 观察

- `ps_bbrpc` 在该单 seed 下 net reward 最高；
- `ce_bbrpc` 触发了 1 次 restart；
- PS planner 的 switching cost 明显高于 CE planner，这符合 posterior sampling 更容易切换 action 的直觉；
- 由于 horizon=3、seed=1、CEM budget 很小，不能据此做统计结论。

---

## 9. 输出文件

```text
reports/tables/brpc_baseline_smoke/config.json
reports/tables/brpc_baseline_smoke/brpc_baseline_raw.csv
reports/tables/brpc_baseline_smoke/brpc_baseline_seed_summary.csv
reports/tables/brpc_baseline_smoke/brpc_baseline_summary.csv
reports/tables/brpc_baseline_smoke/brpc_baseline_report_zh.md
reports/tables/brpc_geometry/toy1_geometry_screening.csv
reports/tables/brpc_geometry/toy2_geometry_screening.csv
reports/tables/brpc_calibration_validation_smoke/calibration_validation.csv
reports/tables/brpc_cem_convergence_smoke/brpc_cem_convergence_raw.csv
reports/tables/brpc_cem_convergence_smoke/brpc_cem_convergence_seed_summary.csv
reports/tables/brpc_cem_convergence_smoke/brpc_cem_convergence_summary.csv
```

---

## 10. Tests 与当前实现状态

当前全量测试通过：

```text
79 passed
```

覆盖内容包括：

- Toy1 / Toy2 dynamics 与 reward accounting；
- BRPC update shape、weight normalization、covariance PSD；
- BOCPD prequential evidence；
- 每个 expert 每步 observation 只 assimilate 一次；
- no-change 不触发 false restart；
- large jump 增加 fresh expert mass；
- pruning 保留 anchor/fresh experts；
- geometry CSV 输出；
- Toy2 diagnostic action geometry gate；
- calibration-only validation schema 与 CLI `--out`；
- PS coherent latent path；
- current/future toy oracle 与 CEM convergence sweep schema。

---

## 11. Caveats

1. 当前是 smoke run，不是论文级实验。
2. Current-dynamics oracle 已接入，future-regime oracle 只作为 toy-only appendix ceiling；二者不能混淆。
3. CEM convergence smoke 显示更大 budget 会提升 oracle realized reward，因此 baseline pilot 不能用过小 CEM budget 做强结论。
4. Toy2 planner 与 geometry 已共享 operating/one-step reward decomposition；正式实验仍需要更高 CEM budget 和 paired seeds。
5. 当前只覆盖 Toy1/Toy2；CartPole BRPC baseline 尚未接入这个新 BRPC suite。
6. Sparse physical data 的正式 claim 需要使用 dual-channel observation：高频 control state $s_t$ 与低频 calibration measurement $y_k^{cal}$ 分离。
7. BOCPD-BRPC 当前是 lightweight implementation；正式实验前还需要 hazard / pruning / restart sensitivity。

---

## 12. 下一步建议

1. 使用通过 oracle convergence smoke 的 CEM budget，先跑 10–20 paired seeds pilot，而不是直接 100+ seeds；
2. 对 Toy2 专门测试 changepoint 后 CE/PS 是否停留在旧 production point，以及 BOCPD restart 是否降低 delay；
3. 对 hazard / pruning / restart margin 做小规模 sensitivity；
4. 如 pilot 显示 benchmark 有区分度，再扩展到 100+ paired seeds；
5. 再接入 CartPole BRPC baseline。
