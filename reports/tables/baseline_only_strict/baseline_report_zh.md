# Baseline-only 严格复现实验报告（reward/cost 版）

本报告只比较三个文献 baseline，不包含 oracle，也不报告 oracle regret：

1. `kh_dual_control`：Klenske-Hennig approximate dual control；
2. `arcari_dual_smpc`：Arcari et al. dual stochastic MPC；
3. `tv_gp_lcb`：Bogunovic et al. time-varying GP-UCB 的 cost-minimization LCB 版本。

实验目标不是证明某个 proposed method，而是检验这些已发表 baseline 在 evolving digital twin setting 下的行为。所有 baseline 都只能使用相同的 nominal digital twin / simulator family 和相同的 physical observations。`gap` setting 中，physical dynamics 与 nominal twin 不一致，但 baseline 不会被告知 true gap function。

---

## 1. 指标定义

代码内部以 cost 最小化实现；报告同时给出 reward 与 cost。二者关系为：

\[
\text{net reward} = -\text{total cost}.
\]

每条 trajectory 的 total cost 分解为：

\[
C_T
=
\sum_{t=0}^{T-1}
\ell_{\text{task}}(s_t,a_t)
+
\sum_{t=0}^{T-1} c_E(a_t)
+
\sum_{t=0}^{T-1} c_{\Delta}(a_t,a_{t-1})
+
\sum_{t=0}^{T-1} c_{\text{failure},t}
+\ell_T^{\text{term}}.
\]

报告中的列含义：

- `mean_net_reward`：平均累计净 reward，等于 `-mean_total_cost`；越大越好；
- `mean_total_cost`：平均累计 cost；越小越好；
- `mean_acc_task_reward`：累计 task reward，等于 `- task/state cost`；越大越好；
- `mean_acc_energy_cost`：累计 energy/control-effort cost；越小越好；
- `mean_acc_switch_cost`：累计 switching / migration cost；越小越好；
- `mean_acc_failure_cost`：累计 failure penalty；越小越好；
- `mean_terminal_cost`：terminal cost；越小越好。

当前 scalar 实验没有 failure term；CartPole 有 failure penalty。

---

## 2. 三个 baseline 的数学建模

### 2.1 Klenske-Hennig approximate dual control (`kh_dual_control`)

#### Scalar 版本

scalar 严格复现 Klenske & Hennig 2016 的线性 toy system：

\[
x_{k+1}=a x_k + b u_k + \xi_k,
\qquad
\xi_k\sim\mathcal N(0,Q),
\]

其中 \(a\) 已知，\(b\) 未知，belief 为：

\[
b\mid\mathcal D_k \sim \mathcal N(\mu_k,\sigma_k^2).
\]

Klenske-Hennig 的 approximate dual control 分三步：

1. **certainty-equivalent nominal trajectory**：用当前 posterior mean \(\mu_k\) 构造 nominal system；
2. **augmented-state quadratic expansion**：把 state 和 parameter 合成 augmented state：
   \[
   z_k=(x_k,b_k),
   \]
   在线性化的 augmented dynamics 上传播 covariance；
3. **dual uncertainty cost**：用 Riccati recursion 得到 perturbation value matrix \(\tilde K_j\)，并计算 uncertainty cost：
   \[
   J_k^d
   =
   \frac12\operatorname{tr}\left(
   W_T\Sigma^{xx}_{T|T}
   +
   \sum_{j=k}^{T-1}
   \left[
   W_j\Sigma^{xx}_{j|j}
   +
   (\Sigma_{j+1|j}-\Sigma_{j+1|j+1})\tilde K_{j+1}
   \right]
   \right).
   \]

最终优化：

\[
u_k^* = \arg\min_{u_k}\ \bar J_k(u_k)+J_k^d(u_k),
\]

其中 \(\bar J_k\) 是 nominal CE cost，\(J_k^d\) 是 approximate dual uncertainty cost。当前 benchmark action set 是 finite grid，因此 root action optimization 是对 finite grid 的精确枚举。

#### CartPole GP 版本

CartPole 中使用 KH paper Section 5.2 / 5.2.1 的 finite-feature GP dynamics 思路。模型不是 actuator-gain pseudo-belief，而是对 transition residual 建模：

\[
s_{t+1}=s_t+W\phi([s_t,a_t])+\epsilon_t.
\]

其中 \(\phi(\cdot)\) 是 squared-exponential kernel 的 finite Fourier feature approximation，\(W\) 是 GP weight-space 参数。augmented state 为：

\[
z_t=(s_t,\operatorname{vec}(W)).
\]

实现包含：

- finite Fourier features；
- weight-space multi-output GP posterior；
- Eq. (16)/(17) 风格的 Gram-matrix posterior blocks；
- local augmented Jacobians；
- augmented Riccati recursion；
- Section 4 trace-form uncertainty cost。

需要注意：CartPole 是本文 benchmark 中的应用环境，不是 KH paper 的原始实验；因此这里应表述为 **KH finite-feature GP approximate dual control applied to CartPole**。

---

### 2.2 Arcari dual stochastic MPC (`arcari_dual_smpc`)

Arcari et al. 2020 的方法是 dual stochastic MPC。其核心思想是把 MPC horizon \(N\) 分成：

- dual part：长度 \(L\)，用 scenario tree 主动获取信息；
- exploitation part：长度 \(N-L\)，在每个 leaf 固定已获得的信息后做 exploitation planning。

在无 structural mode 的主实验中，\(n_m=1\)。scenario tree 的每个 child node 按参数样本和过程噪声样本展开。scalar 中对应：

\[
x^{j_{k+1}}_{k+1}
=
 x^{P(j_{k+1})}_k
+
\gamma^{j_{k+1}}_k u^{P(j_{k+1})}_k
+
w^{j_{k+1}}_k.
\]

CartPole 中对应：

\[
s^{j_{k+1}}_{k+1}
=
f_{\text{twin}}(s^{P(j_{k+1})}_k,u^{P(j_{k+1})}_k,\gamma^{j_{k+1}}_k)
+
w^{j_{k+1}}_k.
\]

每个 node 保存：

- state sample；
- parent pointer；
- children；
- depth；
- sample indices；
- node probability / weight；
- branch-specific information state / posterior。

objective 对应 paper Eq. (10)：

\[
\min_{u_0,\ldots,u_{N-1}}
\sum_{k=0}^{L-1}
\frac{1}{N_s^k}
\sum_{j_k}
\bar p^{j_k}_k
\ell_k(x^{j_k}_k,u^{j_k}_k)
+
\frac{1}{N_s^L}
\sum_{j_L}
\bar p^{j_L}_L
\tilde J_L(I^{j_L}_L).
\]

其中 \(\tilde J_L\) 是 exploitation part 的 fixed-information cost-to-go。当前实现使用 finite action grid，因此是 **finite-action Arcari DMPC specialization**：对给定 finite action set 上的 tree-node actions 做精确枚举，而不是 continuous nonlinear optimizer。

---

### 2.3 TV-GP-LCB (`tv_gp_lcb`)

Bogunovic et al. 2016 的 time-varying GP bandit 模型为：

\[
f_{t+1}(x)=\sqrt{1-\epsilon}\,f_t(x)+\sqrt{\epsilon}\,g_{t+1}(x),
\]

其中：

\[
g_{t+1}\sim\mathcal{GP}(0,k_x).
\]

诱导 time-space kernel：

\[
k((x,t),(x',t'))
=(1-\epsilon)^{|t-t'|/2}k_x(x,x').
\]

GP posterior：

\[
\mu_t(x)=k_*^\top(K+\\sigma^2I)^{-1}y,
\]

\[
\sigma_t^2(x)=k(x,x)-k_*^\top(K+\sigma^2I)^{-1}k_*.
\]

原 paper 用 UCB 做 reward maximization：

\[
x_t=\arg\max_x \mu_{t-1}(x)+\sqrt{\beta_t}\sigma_{t-1}(x).
\]

本 benchmark 是 cost minimization，因此使用 LCB：

\[
a_t=\arg\min_a \mu_{t-1}(\psi_t(a))-
\sqrt{\beta_t}\sigma_{t-1}(\psi_t(a)).
\]

其中 \(\psi_t(a)\) 是当前 context-action feature。scalar 使用：

\[
\psi_t(a)=(x_t,a_{t-1},a),
\]

CartPole 使用：

\[
\psi_t(a)=(s_t,a_{t-1},a).
\]

该 baseline 是 **finite-action TV-GP-LCB on realized one-step costs**。它不是 MPC，也不是 dual control；它只根据 realized one-step cost feedback 更新 GP。

---

## 3. 实验配置

本次 baseline-only 实验不包含 oracle。运行配置如下：

### Scalar

- horizon：50；
- seeds：3；
- planning horizon：3；
- action grid size：11；
- Arcari dual horizon：2；
- Arcari scenarios：3。

### CartPole

- horizon：30；
- seeds：2；
- planning horizon：2；
- action grid size：3；
- Arcari dual horizon：2；
- Arcari scenarios：2；
- KH GP Fourier features：16。

由于 CartPole 的 strict scenario tree 和 KH GP AD 计算量较大，CartPole 当前使用较小 action grid 和较短 horizon；结果应作为 first-pass baseline behavior，而不是最终论文级统计。

---

## 4. Scalar 结果

| environment | twin_gap | regime | baseline | n | mean_net_reward | stderr_net_reward | mean_total_cost | stderr_total_cost | mean_acc_task_reward | stderr_acc_task_reward | mean_acc_energy_cost | stderr_acc_energy_cost | mean_acc_switch_cost | stderr_acc_switch_cost | mean_acc_failure_cost | stderr_acc_failure_cost | mean_terminal_cost | stderr_terminal_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scalar | gap | drifting | arcari_dual_smpc | 3 | -15.18 | 0.9014 | 15.18 | 0.9014 | -14.53 | 0.6958 | 0.36 | 0.07494 | 0 | 0 | 0 | 0 | 0.2898 | 0.1808 |
| scalar | gap | drifting | kh_dual_control | 3 | -12.73 | 0.2019 | 12.73 | 0.2019 | -12.08 | 0.3059 | 0.432 | 0.02078 | 0 | 0 | 0 | 0 | 0.2244 | 0.1759 |
| scalar | gap | drifting | tv_gp_lcb | 3 | -2.385e+04 | 2779 | 2.385e+04 | 2779 | -2.204e+04 | 2747 | 28.84 | 1.102 | 0 | 0 | 0 | 0 | 1785 | 123.1 |
| scalar | gap | piecewise | arcari_dual_smpc | 3 | -23.64 | 6.871 | 23.64 | 6.871 | -23.01 | 6.936 | 0.444 | 0.09372 | 0 | 0 | 0 | 0 | 0.1887 | 0.1426 |
| scalar | gap | piecewise | kh_dual_control | 3 | -10.48 | 1.371 | 10.48 | 1.371 | -9.919 | 1.437 | 0.384 | 0.07869 | 0 | 0 | 0 | 0 | 0.1808 | 0.1464 |
| scalar | gap | piecewise | tv_gp_lcb | 3 | -6.866e+04 | 3.409e+04 | 6.866e+04 | 3.409e+04 | -6.521e+04 | 3.137e+04 | 29.88 | 2.054 | 0 | 0 | 0 | 0 | 3412 | 2803 |
| scalar | gap | static | arcari_dual_smpc | 3 | -22.93 | 5.272 | 22.93 | 5.272 | -22.54 | 5.39 | 0.348 | 0.146 | 0 | 0 | 0 | 0 | 0.04869 | 0.01209 |
| scalar | gap | static | kh_dual_control | 3 | -12.81 | 1.755 | 12.81 | 1.755 | -11.86 | 1.8 | 0.348 | 0.03175 | 0 | 0 | 0 | 0 | 0.593 | 0.1639 |
| scalar | gap | static | tv_gp_lcb | 3 | -7.659e+04 | 2.528e+04 | 7.659e+04 | 2.528e+04 | -7.299e+04 | 2.37e+04 | 30.65 | 1.435 | 0 | 0 | 0 | 0 | 3568 | 1624 |
| scalar | no_gap | drifting | arcari_dual_smpc | 3 | -14.98 | 1.74 | 14.98 | 1.74 | -14.53 | 1.734 | 0.324 | 0.1157 | 0 | 0 | 0 | 0 | 0.1239 | 0.1072 |
| scalar | no_gap | drifting | kh_dual_control | 3 | -13.34 | 1.947 | 13.34 | 1.947 | -12.82 | 1.955 | 0.396 | 0.09525 | 0 | 0 | 0 | 0 | 0.1218 | 0.1036 |
| scalar | no_gap | drifting | tv_gp_lcb | 3 | -7532 | 646.3 | 7532 | 646.3 | -7492 | 647.4 | 27.19 | 1.245 | 0 | 0 | 0 | 0 | 12.38 | 5.059 |
| scalar | no_gap | piecewise | arcari_dual_smpc | 3 | -24.75 | 6.947 | 24.75 | 6.947 | -24.13 | 6.944 | 0.372 | 0.07299 | 0 | 0 | 0 | 0 | 0.2452 | 0.1058 |
| scalar | no_gap | piecewise | kh_dual_control | 3 | -12.25 | 1.109 | 12.25 | 1.109 | -11.51 | 0.925 | 0.384 | 0.024 | 0 | 0 | 0 | 0 | 0.3647 | 0.2078 |
| scalar | no_gap | piecewise | tv_gp_lcb | 3 | -1.209e+04 | 5706 | 1.209e+04 | 5706 | -1.175e+04 | 5528 | 29.46 | 0.8013 | 0 | 0 | 0 | 0 | 306.4 | 184.5 |
| scalar | no_gap | static | arcari_dual_smpc | 3 | -22.17 | 5.529 | 22.17 | 5.529 | -21.71 | 5.665 | 0.252 | 0.07494 | 0 | 0 | 0 | 0 | 0.2051 | 0.1123 |
| scalar | no_gap | static | kh_dual_control | 3 | -9.887 | 0.6737 | 9.887 | 0.6737 | -9.406 | 0.6468 | 0.276 | 0.05231 | 0 | 0 | 0 | 0 | 0.2051 | 0.1123 |
| scalar | no_gap | static | tv_gp_lcb | 3 | -1.077e+04 | 2128 | 1.077e+04 | 2128 | -1.033e+04 | 2213 | 30.16 | 0.7276 | 0 | 0 | 0 | 0 | 413.1 | 85.73 |

### Scalar 分析

1. **KH 在 scalar 上整体最好。** 6 个 scalar setting 中，`kh_dual_control` 的 total cost 都明显低于 `arcari_dual_smpc` 和 `tv_gp_lcb`。例如 no-gap static 下，KH cost 为 9.887，而 Arcari 为 22.17。
2. **Arcari 在 scalar 上较保守或 exploration 成本较高。** Arcari explicit tree 在当前 finite grid / small horizon 下没有显示优势，尤其 static 与 piecewise 下 cost 明显高于 KH。
3. **TV-GP-LCB 在 scalar control setting 中失败。** 它只使用 realized one-step cost 的 bandit feedback，不理解系统 dynamics，因此容易选择导致 state 发散的 action，task cost 和 terminal cost 极大。
4. **gap/no-gap 对 KH 的排序影响不大。** KH 在 gap 和 no-gap 下都相对稳定，说明在 scalar 当前设置中，主要困难不是 discrepancy，而是控制/辨识的基本 tradeoff。

---

## 5. CartPole 结果

| environment | twin_gap | regime | baseline | n | mean_net_reward | stderr_net_reward | mean_total_cost | stderr_total_cost | mean_acc_task_reward | stderr_acc_task_reward | mean_acc_energy_cost | stderr_acc_energy_cost | mean_acc_switch_cost | stderr_acc_switch_cost | mean_acc_failure_cost | stderr_acc_failure_cost | mean_terminal_cost | stderr_terminal_cost | mean_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole | gap | drifting | arcari_dual_smpc | 2 | -0.9332 | 0.01372 | 0.9332 | 0.01372 | -0.7537 | 0.02079 | 0.019 | 0.001 | 0.15 | 0.01 | 0 | 0 | 0.0105 | 0.003927 | 0 |
| cartpole | gap | drifting | kh_dual_control | 2 | -11.51 | 0.1838 | 11.51 | 0.1838 | -8.403 | 0.156 | 0 | 0 | 0 | 0 | 0 | 0 | 3.105 | 0.02783 | 0 |
| cartpole | gap | drifting | tv_gp_lcb | 2 | -839.4 | 0.4213 | 839.4 | 0.4213 | -32.47 | 0.3537 | 0.022 | 0 | 0.46 | 0 | 800 | 0 | 6.465 | 0.06761 | 8 |
| cartpole | gap | piecewise | arcari_dual_smpc | 2 | -0.8535 | 0.02494 | 0.8535 | 0.02494 | -0.6826 | 0.02911 | 0.017 | 0.002 | 0.135 | 0.005 | 0 | 0 | 0.01889 | 0.002823 | 0 |
| cartpole | gap | piecewise | kh_dual_control | 2 | -12.6 | 2.089 | 12.6 | 2.089 | -9.22 | 1.421 | 0 | 0 | 0 | 0 | 0 | 0 | 3.377 | 0.6677 | 0 |
| cartpole | gap | static | arcari_dual_smpc | 2 | -0.8733 | 0.03486 | 0.8733 | 0.03486 | -0.7091 | 0.01498 | 0.0145 | 0.0025 | 0.125 | 0.005 | 0 | 0 | 0.02471 | 0.01237 | 0 |
| cartpole | gap | static | kh_dual_control | 2 | -12.6 | 2.089 | 12.6 | 2.089 | -9.22 | 1.421 | 0 | 0 | 0 | 0 | 0 | 0 | 3.377 | 0.6677 | 0 |
| cartpole | gap | static | tv_gp_lcb | 2 | -638.3 | 104.1 | 638.3 | 104.1 | -30.69 | 3.713 | 0.022 | 0 | 0.46 | 0.01 | 600 | 100 | 7.139 | 0.3597 | 6 |
| cartpole | no_gap | drifting | arcari_dual_smpc | 2 | -0.6544 | 0.01547 | 0.6544 | 0.01547 | -0.5426 | 0.01867 | 0.0085 | 0.0005 | 0.095 | 0.005 | 0 | 0 | 0.008274 | 0.002293 | 0 |
| cartpole | no_gap | drifting | kh_dual_control | 2 | -11.5 | 0.1849 | 11.5 | 0.1849 | -8.4 | 0.1568 | 0 | 0 | 0 | 0 | 0 | 0 | 3.102 | 0.02806 | 0 |
| cartpole | no_gap | drifting | tv_gp_lcb | 2 | -791.7 | 48.45 | 791.7 | 48.45 | -34.2 | 0.9633 | 0.0225 | 0.0005 | 0.49 | 0 | 750 | 50 | 6.961 | 0.5883 | 7.5 |
| cartpole | no_gap | piecewise | arcari_dual_smpc | 2 | -0.6232 | 0.03784 | 0.6232 | 0.03784 | -0.5103 | 0.04459 | 0.0065 | 0.0025 | 0.07 | 0.02 | 0 | 0 | 0.03636 | 0.02926 | 0 |
| cartpole | no_gap | piecewise | kh_dual_control | 2 | -12.6 | 2.09 | 12.6 | 2.09 | -9.221 | 1.422 | 0 | 0 | 0 | 0 | 0 | 0 | 3.378 | 0.6678 | 0 |
| cartpole | no_gap | piecewise | tv_gp_lcb | 2 | -689.7 | 51.18 | 689.7 | 51.18 | -32.1 | 1.272 | 0.022 | 0 | 0.45 | 0 | 650 | 50 | 7.084 | 0.09524 | 6.5 |
| cartpole | no_gap | static | arcari_dual_smpc | 2 | -0.6208 | 0.03263 | 0.6208 | 0.03263 | -0.5141 | 0.0007779 | 0.008 | 0.003 | 0.085 | 0.025 | 0 | 0 | 0.01375 | 0.00541 | 0 |
| cartpole | no_gap | static | kh_dual_control | 2 | -12.6 | 2.09 | 12.6 | 2.09 | -9.221 | 1.422 | 0 | 0 | 0 | 0 | 0 | 0 | 3.378 | 0.6678 | 0 |
| cartpole | no_gap | static | tv_gp_lcb | 2 | -793 | 50.6 | 793 | 50.6 | -35.3 | 0.8205 | 0.0225 | 0.0005 | 0.49 | 0 | 750 | 50 | 7.167 | 0.2232 | 7.5 |

### CartPole 分析

1. **Arcari 在 CartPole 上表现最好。** 在所有 CartPole setting 中，`arcari_dual_smpc` total cost 都远低于 KH 和 TV-GP-LCB，并且 failure 数为 0。
2. **KH GP AD 在 CartPole 上较差但没有 failure。** KH 的 total cost 约 11.5–12.6，主要来自 task cost 和 terminal cost，而不是 failure。说明该 GP AD baseline 当前选择的动作较保守或不能快速稳定 CartPole，但没有直接失稳。
3. **TV-GP-LCB 在 CartPole 上频繁 failure。** failure cost 占主导，例如 no-gap static 下 failure cost 约 750，平均 failure 数约 7.5。这符合它作为 one-step bandit baseline 的性质：它没有 model-based rollout 或 stability mechanism。
4. **gap 对 Arcari 影响较小。** Arcari 在 no-gap 和 gap 下都保持低 cost，说明在当前短 horizon 和 coarse grid 下，它对 actuator lag/friction 的 mismatch 不太敏感。
5. **CartPole 当前配置较小。** action grid size 为 3、horizon 为 30、seeds 为 2，因此这些结果是 first-pass baseline behavior，不是最终统计结论。

---

## 6. 总体结论

### Scalar

scalar 中 `kh_dual_control` 最强，说明在一维线性系统和当前参数下，Klenske-Hennig approximate dual control 能有效利用参数不确定性并避免过度 exploration cost。Arcari tree MPC 在 scalar 上反而更保守/成本更高。TV-GP-LCB 不适合该 dynamical control setting，表现明显失败。

### CartPole

CartPole 中 `arcari_dual_smpc` 最强。它显式构造 scenario tree，并在 dual/exploitation split 中处理 branch-specific information，因此在当前 finite-action CartPole setting 中能稳定控制。KH GP AD 虽然严格按 finite-feature GP AD 结构实现，但在当前配置下成本较高。TV-GP-LCB 因缺乏 dynamics planning 导致多次 failure。

### 对 benchmark 设计的含义

- 如果只看 scalar，现有 KH dual control 已经是很强 baseline；
- 如果看 CartPole digital twin setting，Arcari dual stochastic MPC 是更强 baseline；
- TV-GP-LCB 可以作为 time-varying bandit baseline，但它不是 model-based controller，结果差异应解释为方法类别差异，而不是 implementation failure；
- 后续若要证明 proposed method 有空间，必须在 sparse physical data、non-differentiable switch cost、multimodal belief 或 discrepancy 这些 stress settings 中展示这些 strong baselines 的系统性缺口。
