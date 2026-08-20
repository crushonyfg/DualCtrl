# Evolving Digital Twin Baseline Suite：Benchmark、BRPC 与 BOCPD-BRPC 实现规格

> 状态：v1.0，可直接交给实现 agent。  
> 范围：只构建 benchmark、calibration 和 baseline，不包含 proposed planner。  
> 核心问题：在不可重放的 continuous deployment 中，CE 与 posterior sampling 在 gradual drift、abrupt change、低敏感 exploitation 区域和 pairwise switching cost 下会留下多大的累计 physical-return gap？

---

## 1. 第一阶段冻结的实验范围

第一阶段只实现：

1. 两个机制明确的 toy environment；
2. 一个标准非线性控制 benchmark 的 digital-twin/physical-model 实例；
3. 两种 calibration：
   - BRPC（gradual-drift tracking，无 changepoint）；
   - BOCPD-BRPC（B-BRPC，多 restart-time expert）；
4. 两种 planner：
   - certainty equivalent（CE）；
   - posterior sampling（PS）；
5. 一个主 oracle：current-dynamics oracle；
6. 一个只在 toy/附录报告的 future-regime oracle。

由此得到严格的 $2\times2$ baseline：

| Calibration | CE planner | Posterior-sampling planner |
|---|---:|---:|
| BRPC | CE-BRPC | PS-BRPC |
| BOCPD-BRPC | CE-BRPC-CP | PS-BRPC-CP |

第一阶段**不实现**：proposed dual planner、information bonus、UCB/LCB、KH、scenario tree、model-free RL、meta-RL、CVaR、安全约束理论或 joint dynamic regret。

如果这四个 baseline 已经接近 oracle，则先停止方法开发，重新检查 benchmark 是否具有足够的 action-dependent information geometry。

---

## 2. 统一任务定义

### 2.1 Digital twin 与 physical system

控制输入记为 $a_t\in\mathcal A$，可观测 physical state 记为 $s_t\in\mathcal S$。将 calibration 输入统一记为

\[
x_t=(s_t,a_t).
\]

Digital twin 是可重复查询但可能昂贵的 black-box simulator：

\[
\widetilde s_{t+1}
=f_{\mathrm{DT}}(s_t,a_t;\theta_t).
\]

Physical system 只沿真实 trajectory 前进一步：

\[
s_{t+1}
=f_{\mathrm{DT}}(s_t,a_t;\theta_t)
+\delta_t(s_t,a_t)+w_t,
\qquad
w_t\sim\mathcal N(0,\Sigma_w).
\]

其中：

- $\theta_t$ 是随时间演化的 calibration parameter；
- $\delta_t$ 是 simulator discrepancy；
- physical transition 不可为同一个 state 回滚后尝试其他 action；
- simulator rollout 不计入 physical reward，但必须记录 query 数和 wall-clock time；
- online physical data 可以存储并用于 posterior update，但不能将 physical environment replay 成额外交互。

对于 Toy 2（无显式动态状态），令 $s_t=a_{t-1}$，physical observation 为标量 response $y_t$。

### 2.2 唯一 primary objective

所有方法最大化部署期间真实发生的累计净回报：

\[
J_T
=
\sum_{t=0}^{T-1}
\left[
r_{\mathrm{task}}(s_t,a_t,s_{t+1})
-c_E(a_t)
-c_{\mathrm{sw}}(a_{t-1},a_t)
\right]
+r_T^{\mathrm{term}}.
\]

其中

\[
c_E(a_t)=\lambda_E\|a_t\|_2^2,
\qquad
c_{\mathrm{sw}}(a_{t-1},a_t)
=\lambda_\Delta\|a_t-a_{t-1}\|_2^2.
\]

Primary metric 使用 undiscounted realized return，即评估时 $\gamma=1$。planning 内可以使用 $\gamma\le1$，但必须做 sensitivity analysis。

Reward/cost 中不加入 posterior variance、entropy、information gain、UCB/LCB 或 calibration error。信息只有在改善后续真实 physical reward 时才有价值。

### 2.3 统一 belief 内容

单一 BRPC belief 记为

\[
B_t^{\mathrm{BRPC}}
=
\left\{
(\theta_t^{(i)},w_t^{(i)},m_t^{(i)})_{i=1}^N,
C_t,
Z,
\phi
\right\},
\]

其中：

- $N$：参数粒子数；
- $Z=\{z_1,\ldots,z_M\}$：固定 inducing/support set；
- $m_t^{(i)}$：给定参数粒子 $i$ 的 discrepancy inducing mean；
- $C_t$：所有粒子共享的 discrepancy covariance；
- $\phi$：冻结的 GP/kernel/noise hyperparameters。

BOCPD-BRPC belief 是 BRPC expert mixture：

\[
B_t^{\mathrm{CP}}
=
\left\{
(\alpha_{e,t},s_e,B_{e,t}^{\mathrm{BRPC}})
:e\in\mathcal E_t
\right\},
\qquad
\sum_{e\in\mathcal E_t}\alpha_{e,t}=1,
\]

其中 $s_e$ 是 expert 假定的当前 segment start time。

---

## 3. Benchmark 进入正式实验前的 geometry gate

在实现全部 baseline 之前，必须用 ground-truth physical model 对每个环境做 geometry screening。环境只有同时满足以下条件才通过：

### G1：参数变化具有决策相关性

存在 $\theta^{\mathrm{old}},\theta^{\mathrm{new}}$，使得

\[
a^\star(\theta^{\mathrm{old}})
\neq
a^\star(\theta^{\mathrm{new}}),
\]

并且使用旧 action 在新 regime 中产生显著累计损失。

### G2：旧 exploitation 区域低敏感

在旧最优 action/state 附近：

\[
\mathcal I_\theta(s,a)
\approx0
\quad\text{或}\quad
D_{\mathrm{KL}}
\left(
p_{\mathrm{new}}(y\mid s,a)
\Vert
p_{\mathrm{old}}(y\mid s,a)
\right)
\approx0.
\]

### G3：存在有代价的 diagnostic 区域

存在 $a^{\mathrm{diag}}$，使其参数敏感度或 old/new predictive KL 显著更高，但 immediate reward 更低或 switching cost 更高。

### G4：诊断信息需要多步才能转化为 reward

最短需要至少两步：

\[
a_t^{\mathrm{diag}}
\rightarrow y_{t+1}
\rightarrow a_{t+1}(y_{t+1}).
\]

若还要从生产点经过诊断点再迁移到新生产点，应至少需要 $M\ge3$。

### G5：posterior sampling 不会自然覆盖所有诊断行为

Toy 2 要满足更强条件：

\[
a^{\mathrm{diag}}
\notin
\bigcup_{\theta\in\Theta}
\arg\max_a r(a;\theta).
\]

即 diagnostic action 对任何单一 sampled model 都不是生产最优，但可能具有正的 Bayes multi-step value。

### 必画的四张图

对 state/action grid 画：

1. expected physical reward；
2. 参数敏感度、经验 Fisher information 或 predictive variance reduction；
3. old/new predictive KL；
4. 从当前 action 出发的 switching cost。

如果 reward optimum、information optimum 和 change-detection optimum 基本重合，则该参数配置不能作为主实验。

---

## 4. Toy 1：Delayed-Excitation Dynamics

### 4.1 目的

验证：

- $M=1$ 看不到 calibration value；
- drift 下低激励 exploitation 使 posterior stale；
- probe-out-and-return 需要显式计算 pairwise switching cost；
- fixed nonzero discrepancy 下仍需分离 $\theta_t$ 与 $\delta_t$。

### 4.2 Digital twin 与 physical system

状态、动作均为标量：

\[
x_t\in\mathbb R,
\qquad
a_t\in[-1,1].
\]

Digital twin：

\[
f_{\mathrm{DT}}(x_t,a_t;\theta_t)
=\theta_t x_t+a_t.
\]

Physical system：

\[
x_{t+1}
=\theta_t x_t+a_t
+\beta_t\tanh(2x_t)
+\kappa_\delta a_t|a_t|
+w_t,
\qquad
w_t\sim\mathcal N(0,\sigma_w^2).
\]

BRPC 的 calibration input 为

\[
X_t=(x_t,a_t),
\qquad
Y_t=x_{t+1}.
\]

参数敏感度为

\[
\frac{\partial f_{\mathrm{DT}}}{\partial\theta_t}=x_t.
\]

在 $x_t=0$ 时，选择 $a_t\neq0$ 只能先生成 $x_{t+1}=a_t+\text{noise}$；直到下一次 transition 中的 $\theta_t x_{t+1}$ 才提供明显参数信息。因此 $M=1$ 结构性不足，$M\ge2$ 才能看到信息产生，$M\ge3$ 才能完整评价 probe-and-return。

### 4.3 Reward/cost

\[
r_t
=
-q_x(x_t-x_t^{\mathrm{ref}})^2
-\lambda_Ea_t^2
-\lambda_\Delta(a_t-a_{t-1})^2.
\]

参考轨迹使用已知的 phase schedule：

\[
x_t^{\mathrm{ref}}
=
\begin{cases}
0, & \text{maintenance/quiet phase},\\
r_{\mathrm{prod}}, & \text{tracking phase}.
\end{cases}
\]

主实验中 phase schedule 对所有 planner 已知。quiet phase 的即时最优行为接近 $a=0$，但未来 tracking phase 对 $\theta_t$ 的准确度敏感。

### 4.4 初始配置（pilot 后可调）

```yaml
toy1:
  horizon_T: 400
  planning_horizon_M: [1, 2, 3, 5, 10]
  theta_range: [0.60, 1.25]
  theta_initial: 0.85
  beta_initial: 0.08
  kappa_delta: 0.03
  sigma_w: 0.03
  q_x: 1.0
  lambda_energy: 0.05
  lambda_switch: 0.20
  action_range: [-1.0, 1.0]
  cold_start_transitions: 50
```

正式参数必须通过第 3 节 geometry gate；以上数值不是最终结论的一部分。

---

## 5. Toy 2：Stealth Changepoint Operating Landscape

### 5.1 目的

验证：

- changepoint 已发生，但旧生产点的 observation 几乎不变；
- BOCPD 可能已有弱怀疑，但 threshold 尚未触发；
- CE 停留在旧生产点；
- PS 只会选择某个 sampled model 的生产最优点，不主动选择“从不生产最优”的诊断点；
- 中间诊断点同时具有信息价值和 switching-route 价值。

### 5.2 Response model

动作是连续生产配置：

\[
a_t\in[0,1].
\]

定义三个中心：

\[
a_L=0.2,
\qquad
a_D=0.5,
\qquad
a_R=0.8,
\]

以及 Gaussian basis：

\[
\phi_\mu(a)
=
\exp\left[-\frac{(a-\mu)^2}{2\sigma_a^2}\right].
\]

Digital twin response：

\[
f_{\mathrm{DT}}(a;\theta)
=
b_L\phi_{a_L}(a)
+(b_R+c_R\theta)\phi_{a_R}(a)
+(b_D+c_D\theta)\phi_{a_D}(a).
\]

Physical response：

\[
y_t
=
f_{\mathrm{DT}}(a_t;\theta_t)
+\delta_t(a_t)
+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma_y^2).
\]

第一阶段可以使用固定但非零 discrepancy：

\[
\delta(a)=A_\delta\sin(4\pi a)+B_\delta(a-0.5)^3.
\]

### 5.3 Reward 与 switching geometry

Physical response 直接视为生产收益：

\[
r_t
=
y_t
-\lambda_Ea_t^2
-\lambda_\Delta(a_t-a_{t-1})^2.
\]

初始参数可设为：

\[
(b_L,b_R,b_D)=(1.2,0.7,-0.4),
\qquad
(c_R,c_D)=(0.9,1.5),
\qquad
\sigma_a=0.08.
\]

在 $\theta$ 从低值变为高值时：

- 左侧 $a_L$ 从旧生产最优变为次优；
- 右侧 $a_R$ 变为新生产最优；
- 中间 $a_D$ 对 $\theta$ 最敏感，但通过降低 $b_D$ 保证它在整个 $\Theta$ 内都不是单模型生产最优。

必须数值验证：

\[
\min_{\theta\in\Theta}
\left[
\max_a r(a;\theta)-r(a_D;\theta)
\right]
\ge \varepsilon_{\mathrm{prod}}>0,
\]

以及

\[
\mathcal I_\theta(a_D)
>
\max\{\mathcal I_\theta(a_L),\mathcal I_\theta(a_R)\}.
\]

由于 $a_D$ 位于 $a_L,a_R$ 之间，平方 switching cost 产生：

\[
(a_D-a_L)^2+(a_R-a_D)^2
<
(a_R-a_L)^2.
\]

因此诊断点可以成为由旧配置向新配置移动的 bridge，但在没有 change 时仍可能需要返回并支付真实成本。

### 5.4 初始配置

```yaml
toy2:
  horizon_T: 300
  planning_horizon_M: [1, 2, 3, 5, 10]
  theta_range: [0.0, 1.0]
  theta_initial: 0.10
  theta_after_jump: 1.00
  sigma_basis: 0.08
  b_left: 1.20
  b_right: 0.70
  b_diag: -0.40
  c_right: 0.90
  c_diag: 1.50
  discrepancy_sine_amplitude: 0.05
  sigma_y: 0.03
  lambda_energy: 0.05
  lambda_switch: 0.80
  cold_start_transitions: 80
```

---

## 6. Benchmark：Continuous CartPole Digital Twin

### 6.1 定位

CartPole 是广泛使用的非线性控制/RL benchmark。本实验将其**实例化**为 nominal digital twin 与 hidden high-fidelity physical plant 的配对环境。不要在论文中声称 CartPole 本身是统一公认的 digital-twin benchmark。

状态与连续动作：

\[
s_t=(p_t,\dot p_t,\varphi_t,\dot\varphi_t),
\qquad
a_t\in[-a_{\max},a_{\max}],
\]

其中 $\varphi=0$ 表示 upright。

### 6.2 Nominal digital-twin dynamics

使用标准 CartPole dynamics 和固定 Euler/semi-implicit Euler 离散化。给定 applied force $F_t$：

\[
\mathrm{temp}
=
\frac{F_t+m_p\ell\dot\varphi_t^2\sin\varphi_t}
{m_c+m_p},
\]

\[
\ddot\varphi_t
=
\frac{
g\sin\varphi_t
-\cos\varphi_t\,\mathrm{temp}
}{
\ell\left(
\frac43-
\frac{m_p\cos^2\varphi_t}{m_c+m_p}
\right)
},
\]

\[
\ddot p_t
=
\mathrm{temp}
-
\frac{m_p\ell\ddot\varphi_t\cos\varphi_t}{m_c+m_p}.
\]

Digital twin 的 applied force 为

\[
F_t^{\mathrm{DT}}
=k_{m,t}a_t-b_{v,t}\dot p_t,
\]

calibration parameter 先限制为

\[
\theta_t=(k_{m,t},b_{v,t},m_{p,t}).
\]

Planner 和 calibration 只能通过 simulator API 查询：

```text
next_state = digital_twin.step(state, action, theta)
```

不得使用 dynamics derivative。

### 6.3 Hidden physical system

Physical plant 增加 simulator 未显式建模的 actuator/friction dynamics：

\[
v_t^{\mathrm{cmd}}
=
\operatorname{deadzone}(a_t;d_t),
\]

\[
F_t
=
(1-\alpha_F)F_{t-1}
+\alpha_F
\left[
k_{m,t}v_t^{\mathrm{cmd}}
-b_{v,t}\dot p_t
-f_{c,t}\tanh(\dot p_t/v_0)
\right],
\]

再将 $F_t$ 输入同一刚体方程，并可加入：

- force saturation；
- one-step actuator delay；
- rail-position-dependent friction；
-小幅 process noise。

BRPC discrepancy 学习的是 resulting next-state residual：

\[
\delta_t(s_t,a_t)
=
s_{t+1}^{\mathrm{phys}}
-f_{\mathrm{DT}}(s_t,a_t;\theta_t).
\]

四个 state output 第一阶段使用四个相互独立、共享输入 kernel 的 scalar GP；不要一开始实现 full multi-output GP。

数值实现中优先令 calibration target 为标准化 state increment：

\[
Y_t
=
\operatorname{Std}
\left(
s_{t+1}-s_t
\right),
\]

digital twin 输出同样转换成

\[
y_s(X_t,\theta)
=
\operatorname{Std}
\left(
f_{\mathrm{DT}}(s_t,a_t;\theta)-s_t
\right).
\]

这里 `Std` 使用 cold-start statistics，并在 evaluation 中冻结。这样可以避免绝对 position/angle scale 支配 likelihood，也使 velocity/acceleration channel 中的参数信息更明显。

### 6.4 Task 和 realized reward

任务是 upright stabilization 加已知 cart-position reference tracking：

\[
r_t
=-
\left[
w_p(p_t-p_t^{\mathrm{ref}})^2
+w_\varphi\operatorname{wrap}(\varphi_t)^2
+w_v\dot p_t^2
+w_\omega\dot\varphi_t^2
+\lambda_Ea_t^2
+\lambda_\Delta(a_t-a_{t-1})^2
\right]
-C_{\mathrm{fail}}\mathbf1\{\mathrm{failure}\}.
\]

Reference 每隔一个 block 在 $\{0,+p_0,0,-p_0\}$ 之间平滑改变。该 schedule 对所有 planner 已知。

Failure 初始定义：

\[
|p_t|>2.4
\quad\text{或}\quad
|\varphi_t|>20^\circ.
\]

首选处理方式是进入 absorbing failure state，并对剩余 horizon 每步计 downtime cost。若工程上必须 reset，则 reset 必须：

1. 对所有方法规则完全相同；
2. 计固定 maintenance/downtime penalty；
3. 不清空 calibration history，除非 detector 自己触发 restart。

### 6.5 参数演化

Gradual drift：

- $k_{m,t}$ 缓慢下降；
- $b_{v,t}$ 缓慢上升；
- physical discrepancy 保持固定或做很小的 inducing-state drift。

Abrupt change：

- $m_{p,t}$ 因 payload 突然增加；或
- motor deadzone $d_t$ / Coulomb friction $f_{c,t}$ 突然改变。

前者主要改变 $\theta_t$，后者主要改变 $\delta_t$。第一阶段主结果分别改变一个通道，joint $\theta+\delta$ change 只作为 stress test。

### 6.6 初始配置

```yaml
cartpole:
  dt: 0.02
  horizon_T: 2000
  planning_horizon_M: [10, 20, 30]
  action_range: [-10.0, 10.0]
  mass_cart: 1.0
  mass_pole_initial: 0.10
  half_length: 0.50
  gravity: 9.8
  motor_gain_initial: 1.0
  viscous_friction_initial: 0.05
  actuator_lag_alpha: 0.30
  coulomb_friction: 0.05
  motor_deadzone: 0.20
  w_position: 1.0
  w_angle: 20.0
  w_velocity: 0.10
  w_angular_velocity: 0.10
  lambda_energy: 0.001
  lambda_switch: 0.01
  cold_start_transitions: 500
```

---

## 7. Cold start 与 evolving regimes

### 7.1 Cold-start protocol

Cold-start physical dataset 在 online evaluation 之前一次性生成，所有方法共享完全相同的数据和顺序。Cold-start reward 不计入主 online return，但必须报告数据量。

建议：

| Environment | 主设置 | sensitivity sweep |
|---|---:|---:|
| Toy 1 | 50 transitions | 25 / 50 / 100 |
| Toy 2 | 80 transitions | 25 / 50 / 100 |
| CartPole | 500 transitions | 200 / 500 / 1000 |

Cold-start policy：

- Toy 1：安全反馈控制器 + 小幅 chirp；
- Toy 2：覆盖 $[0,1]$ 的 space-filling design，但只在 initial regime 采集；
- CartPole：nominal LQR/PD stabilizer + 小幅 PRBS/chirp perturbation。

Cold start 必须使系统大致可控，但 posterior 不能接近退化。报告 cold-start 后的 particle variance、predictive RMSE 和 coverage。

### 7.2 Gradual drift

使用有界 OU / AR$1$：

\[
\theta_{t+1}
=
\Pi_\Theta\left[
\bar\theta
+\rho_\theta(\theta_t-\bar\theta)
+L_\theta\varepsilon_t
\right],
\qquad
\varepsilon_t\sim\mathcal N(0,I).
\]

正式实验使用随机轨迹；线性或正弦 drift 只用于可视化 sanity check。

### 7.3 Abrupt change

Change time 不固定给 planner：

\[
\tau\sim\operatorname{Uniform}
(0.35T,0.65T),
\]

或者每步使用 hazard $h$，并设置 minimum segment length。发生 change 时：

\[
\theta_\tau\sim\pi_0^{\mathrm{new}}
\]

或从预先声明的 truncated jump distribution 采样。不要让 planner 知道 realized $\tau$，但 BOCPD-BRPC 可以知道正确或轻度 misspecified 的 hazard。

### 7.4 配对随机数

同一个 evaluation seed 下，所有方法必须共享：

- $\theta_{0:T}$；
- changepoint time 和 jump size；
- physical process/observation noise；
- reference schedule；
- cold-start dataset。

方法内部的 posterior/planner random seed 单独记录。

---

## 8. BRPC-F 的具体数学与实现

本节规定第一阶段唯一实现的 BRPC 版本：**固定 support、参数粒子、particle-specific discrepancy mean、shared discrepancy covariance**。这对应可扩展的 BRPC-F；不要同时实现 BRPC-E/P/RRA。

### 8.1 Projected-calibration target

概念上，calibration target 定义为：

\[
\theta_t
\in
\arg\min_{\theta\in\Theta}
\int
\left\{
\zeta_t(x)-y_s(x,\theta)
\right\}^2
\,dF_{\mathrm{ref}}(x).
\]

然后条件 discrepancy 为

\[
\delta_t(x)=\zeta_t(x)-y_s(x,\theta_t).
\]

在 controlled deployment 中 $x_t=(s_t,a_t)$ 是 policy-selected，而不是 i.i.d. design。第一版实现仍按条件 likelihood 更新，但必须：

1. 固定并记录 $F_{\mathrm{ref}}$；
2. cold start 覆盖该参考区域；
3. 报告 online visitation coverage；
4. 不随不同 planner 重新定义 $F_{\mathrm{ref}}$。

这是第一版的建模假设，不要暗中把 calibration target 改成每个 policy 自己的 occupancy projection。

### 8.2 参数 predictive prior

给定上一时刻粒子：

\[
q_{t-1}^{\theta}
\approx
\sum_{i=1}^Nw_{t-1}^{(i)}
\delta_{\theta_{t-1}^{(i)}}.
\]

使用 drift transition：

\[
\theta_{t\mid t-1}^{(i)}
\sim
p(\theta_t\mid\theta_{t-1}^{(i)}),
\]

例如：

\[
\theta_{t\mid t-1}^{(i)}
=
\Pi_\Theta\left[
\bar\theta
+\rho_\theta(\theta_{t-1}^{(i)}-\bar\theta)
+L_\theta\epsilon_t^{(i)}
\right].
\]

预测权重保持：

\[
w_{t\mid t-1}^{(i)}=w_{t-1}^{(i)}.
\]

### 8.3 Discrepancy-free projected particle update

对 incoming physical batch $(X_t,Y_t)$，参数更新**故意不加入 discrepancy**：

\[
p_{\mathrm{proj}}
(Y_t\mid X_t,\theta)
=
\mathcal N
\left(
Y_t;
y_s(X_t,\theta),
\Sigma_{\theta,t}
\right).
\]

KL-regularized update：

\[
q_t^\theta
=
\arg\min_q
\left\{
-\eta_\theta
\mathbb E_q
\log p_{\mathrm{proj}}(Y_t\mid X_t,\theta)
+\mathrm{KL}(q\Vert\bar q_{t\mid t-1}^\theta)
\right\},
\]

其解为 tempered posterior：

\[
q_t^\theta(\theta)
\propto
\bar q_{t\mid t-1}^\theta(\theta)
p_{\mathrm{proj}}(Y_t\mid X_t,\theta)^{\eta_\theta}.
\]

粒子 log weight：

\[
\log\widetilde w_t^{(i)}
=
\log w_{t\mid t-1}^{(i)}
+\eta_\theta
\log p_{\mathrm{proj}}
(Y_t\mid X_t,\theta_{t\mid t-1}^{(i)}).
\]

必须使用 `logsumexp` 归一化。

Effective sample size：

\[
\mathrm{ESS}_t
=
\left(
\sum_i(w_t^{(i)})^2
\right)^{-1}.
\]

若

\[
\mathrm{ESS}_t<\tau_{\mathrm{ESS}}N,
\]

使用 systematic/stratified resampling。Resampling 时必须将对应的 discrepancy mean $m_t^{(i)}$ 与参数粒子一起复制；不能只 resample $\theta$。Shared covariance $C_t$ 不需要复制。

### 8.4 Fixed-support GP discrepancy state

固定 inducing/support set：

\[
Z=\{z_1,\ldots,z_M\},
\qquad
u_t=\delta_t(Z).
\]

GP prior kernel matrix：

\[
K_{ZZ}=\left[k_\phi(z_i,z_j)\right]_{i,j=1}^M.
\]

建议首版使用 ARD RBF kernel：

\[
k_\phi(x,x')
=
\sigma_\delta^2
\exp\left[
-\frac12
\sum_d
\frac{(x_d-x_d')^2}{\ell_d^2}
\right].
\]

输入 $X$ 必须标准化；kernel hyperparameters 只在 cold-start/validation data 上选择，evaluation 中冻结。

### 8.5 Discrepancy temporal prior

采用论文中的 Gaussian Markov evolution：

\[
u_t
=
\rho_\delta u_{t-1}
+\sqrt{1-\rho_\delta^2}\,\xi_t,
\qquad
\xi_t\sim\mathcal N(0,K_{ZZ}).
\]

若上一 posterior 为

\[
u_{t-1}^{(i)}\mid\mathcal D_{t-1}
\sim
\mathcal N(m_{t-1}^{(i)},C_{t-1}),
\]

则 pre-update prior 为

\[
a_t^{(i)}
=\rho_\delta m_{t-1}^{(i)},
\]

\[
P_t
=
\rho_\delta^2C_{t-1}
+(1-\rho_\delta^2)K_{ZZ}
+Q_\delta.
\]

$Q_\delta\succeq0$ 是可选 covariance inflation/process-noise term。$P_t$ 对所有粒子共享。

### 8.6 Support-to-batch map

对当前 input batch $X_t$：

\[
G_t
=
K_{X_tZ}K_{ZZ}^{-1},
\]

\[
Q_t^{F}
=
K_{X_tX_t}
-K_{X_tZ}K_{ZZ}^{-1}K_{ZX_t}.
\]

不要显式求逆；用带 jitter 的 Cholesky solve。

将 inducing approximation variance 合入 residual covariance：

\[
R_t^{\mathrm{eff}}
=
\Sigma_{\epsilon,t}
+Q_t^{F}
+\sigma_{\mathrm{nug}}^2I.
\]

### 8.7 Conditional discrepancy update

给定更新后的参数粒子，particle-specific residual：

\[
r_t^{(i)}
=
Y_t-y_s(X_t,\theta_t^{(i)}).
\]

Residual likelihood：

\[
r_t^{(i)}
=G_tu_t^{(i)}+\varepsilon_t,
\qquad
\varepsilon_t\sim
\mathcal N(0,R_t^{\mathrm{eff}}).
\]

KL-regularized Gaussian update：

\[
q_t^{\delta,(i)}
=
\arg\min_q
\left\{
-\eta_\delta
\mathbb E_q\log p(r_t^{(i)}\mid u_t^{(i)})
+\mathrm{KL}
(q\Vert\mathcal N(a_t^{(i)},P_t))
\right\}.
\]

Posterior covariance 对所有粒子共享：

\[
C_t^{-1}
=
P_t^{-1}
+\eta_\delta
G_t^\top(R_t^{\mathrm{eff}})^{-1}G_t.
\]

Posterior mean：

\[
m_t^{(i)}
=
C_t
\left[
P_t^{-1}a_t^{(i)}
+\eta_\delta
G_t^\top(R_t^{\mathrm{eff}})^{-1}r_t^{(i)}
\right].
\]

等价地：

\[
m_t^{(i)}
=
\arg\min_u
\left\{
\frac{\eta_\delta}{2}
\|r_t^{(i)}-G_tu\|_{(R_t^{\mathrm{eff}})^{-1}}^2
+\frac12
\|u-a_t^{(i)}\|_{P_t^{-1}}^2
\right\}.
\]

实现中全部使用 Cholesky/information-form solve，不形成 $P^{-1},R^{-1},C^{-1}$ 的显式 inverse。

### 8.8 Predictive distribution

对新 inputs $X_\star$：

\[
G_\star
=K_{X_\star Z}K_{ZZ}^{-1},
\]

\[
Q_\star
=K_{X_\star X_\star}
-K_{X_\star Z}K_{ZZ}^{-1}K_{ZX_\star}.
\]

给定粒子 $i$：

\[
\mu_{\star,t}^{(i)}
=
y_s(X_\star,\theta_t^{(i)})
+G_\star m_t^{(i)},
\]

\[
\Sigma_{\star,t}^{(i)}
=
\Sigma_s(X_\star,\theta_t^{(i)})
+Q_\star
+G_\star C_tG_\star^\top
+\Sigma_{\epsilon,\star}.
\]

若 simulator deterministic，则 $\Sigma_s=0$。整体 predictive law：

\[
p_t(Y_\star\mid X_\star)
=
\sum_{i=1}^N
w_t^{(i)}
\mathcal N
(Y_\star;\mu_{\star,t}^{(i)},\Sigma_{\star,t}^{(i)}).
\]

### 8.9 多输出 transition

对于 $Y_t=s_{t+1}\in\mathbb R^{d_s}$，首版实现：

- 共享 $\theta$ 粒子；
- 每个 output dimension $j$ 单独维护 $m_{t,j}^{(i)},C_{t,j},K_{ZZ,j}$；
- 条件独立 predictive likelihood：

\[
\log p(Y_t\mid X_t,\theta^{(i)})
=
\sum_{j=1}^{d_s}
\log p(Y_{t,j}\mid X_t,\theta^{(i)}).
\]

在使用 BOCPD 时必须对不同 state dimensions 做标准化，否则某一高尺度维度会支配 changepoint evidence。

### 8.10 BRPC 单步伪代码

```text
BRPC_STEP(state, X_t, Y_t):
    # 1. Predict theta and discrepancy states
    theta_pred[i] ~ p(theta_t | theta_prev[i])
    a[i] = rho_delta * m_prev[i]
    P = rho_delta^2 * C_prev + (1-rho_delta^2) * K_ZZ + Q_delta

    # 2. Projected/discrepancy-free theta update
    logw[i] = log(w_prev[i])
              + eta_theta * logN(Y_t; twin(X_t, theta_pred[i]), Sigma_theta)
    w = softmax_logweights(logw)

    # 3. Conditional discrepancy update
    G, QF = gp_support_map(X_t, Z)
    R_eff = Sigma_eps + QF + nugget
    C = inverse_information(P, G, R_eff, eta_delta)
    for i:
        residual[i] = Y_t - twin(X_t, theta_pred[i])
        m[i] = gaussian_information_mean(P, a[i], G, R_eff,
                                         residual[i], eta_delta)

    # 4. Coupled resampling
    if ESS(w) < ess_fraction * N:
        idx = systematic_resample(w)
        theta = theta_pred[idx]
        m = m[idx]
        w = uniform_weights(N)
    else:
        theta = theta_pred

    return updated_state(theta, w, m, C)
```

### 8.11 Inducing support 与初始数值配置

Inducing inputs 必须在标准化后的 calibration-input space 中选择并冻结：

- Toy 1：从 cold-start $X$ 加 action/state grid 后做 k-means，$M_Z=24$；
- Toy 2：在 $[0,1]$ 上使用等距 grid，$M_Z=32$；
- CartPole：对 cold-start $(s,a)$ 做 k-means，$M_Z=64$；
- CartPole 四个 output GP 共用 $Z$，但允许不同 kernel output scale 和 noise variance。

第一轮起始配置：

```yaml
brpc:
  num_theta_particles_toy: 128
  num_theta_particles_cartpole: 256
  ess_fraction: 0.50
  eta_theta: 1.0
  eta_delta: 1.0
  rho_theta: 0.995
  rho_delta: 0.995
  inducing_points_toy1: 24
  inducing_points_toy2: 32
  inducing_points_cartpole: 64
  covariance_jitter: 1.0e-6
  covariance_inflation: 1.0e-5
```

$\eta_\theta,\eta_\delta,\Sigma_\theta,\Sigma_\epsilon$ 和 kernel hyperparameters 只能在 calibration/validation seeds 上选择。特别是 discrepancy-free parameter likelihood 不能使用过小的 $\Sigma_\theta$，否则固定 simulator gap 会导致粒子退化。正式报告至少包含 $\eta_\theta,\eta_\delta\in\{0.25,0.5,1.0\}$ 的局部 sensitivity。

---

## 9. BOCPD-BRPC（B-BRPC）的具体实现

### 9.1 Expert state

每个 expert $e$ 保存：

```text
expert.start_time = s_e
expert.log_mass = log(alpha_e)
expert.brpc_state = {theta_particles, theta_weights,
                     discrepancy_means, discrepancy_covariances,
                     inducing_support, frozen_hyperparameters}
```

Online deployment 开始时：

- anchor expert 使用 cold-start 后的 BRPC posterior；
- 后续 fresh expert 使用 restart prior，而不是 cold-start posterior。

### 9.2 Restart prior

参数：

\[
\theta^{(i)}\sim\pi_0^\theta,
\qquad
w^{(i)}=1/N.
\]

若没有可靠工程 prior，使用 admissible range 上的 uniform distribution。

Discrepancy：

\[
u^{(i)}\sim\mathcal N(0,K_{ZZ}),
\qquad
m_0^{(i)}=0,
\qquad
C_0=K_{ZZ}.
\]

Fresh expert 对 incoming batch 的 prior predictive：

\[
p_{\mathrm{new}}^{\mathrm{pre}}(Y_t\mid X_t)
=
\frac1N
\sum_{i=1}^N
\mathcal N
\left(
Y_t;
y_s(X_t,\theta^{(i)}),
Q_{\star,0}+G_{\star,0}K_{ZZ}G_{\star,0}^\top
+\Sigma_\epsilon
\right).
\]

使用 fixed-support exact formula 时，上式 GP covariance 可直接写成 $K_{X_tX_t}+\Sigma_\epsilon$。

### 9.3 Pre-update predictive evidence

对 continuation expert，必须先执行 propagation、但不能 assimilate 当前 $Y_t$：

1. propagate $\theta_{t-1}^{(i)}\to\theta_{t\mid t-1}^{(i)}$；
2. propagate $m_{t-1}^{(i)},C_{t-1}\to a_t^{(i)},P_t$；
3. 用 predictive mixture 计算

\[
p_e^{\mathrm{pre}}(Y_t\mid X_t)
=
\sum_iw_{e,t-1}^{(i)}
\mathcal N
(Y_t;\mu_{e,t}^{(i),-},\Sigma_{e,t}^{(i),-}).
\]

必须缓存 propagated state，后续 expert update 复用，避免同一步随机传播两次。

Likelihood 在 log domain 中计算：

\[
\log p_e^{\mathrm{pre}}
=
\operatorname{logsumexp}_i
\left[
\log w_{e,t-1}^{(i)}
+\log\mathcal N
(Y_t;\mu_{e,t}^{(i),-},\Sigma_{e,t}^{(i),-})
\right].
\]

### 9.4 BOCPD expert-weight recursion

给定 hazard $h_t$：

Continuation expert：

\[
\widetilde\alpha_{e,t}
=(1-h_t)\alpha_{e,t-1}
p_e^{\mathrm{pre}}(Y_t\mid X_t).
\]

Fresh restart expert：

\[
\widetilde\alpha_{\mathrm{new},t}
=h_t
p_{\mathrm{new}}^{\mathrm{pre}}(Y_t\mid X_t).
\]

归一化：

\[
\alpha_{e,t}
=
\frac{\widetilde\alpha_{e,t}}
{\sum_{e'}\widetilde\alpha_{e',t}}.
\]

同样必须在 log domain 中完成。

### 9.5 Hard-anchor restart rule

令 $e_t^{\mathrm{anc}}$ 是最近一次 hard restart 选择的 anchor。论文式 restart rule：

\[
\operatorname{Restart}_t
=
\mathbf1
\left\{
\max_{e:s_e>s_{e_t^{\mathrm{anc}}}}
\alpha_{e,t}
>
\rho_B\alpha_{e_t^{\mathrm{anc}},t}
\right\},
\qquad
\rho_B\ge1.
\]

若触发：

1. 选择 posterior mass 最大的 post-anchor expert 为新 anchor；
2. 丢弃 start time 早于新 anchor 的 experts；
3. 保留新 anchor 及其后的候选 experts；
4. 记录 restart event；
5. 不重新用同一数据再更新一次。

初始设置 $\rho_B=1$。

### 9.6 Expert pruning

每步最多保留 $M_{\max}$ 个 expert：

\[
\mathcal E_t
=
\operatorname{Top}_{M_{\max}}
(\mathcal E_t^{\mathrm{cand}};\alpha_{e,t}).
\]

必须总是保留：

- 当前 anchor；
- 当前 fresh expert；
- 其余按 posterior mass 取 top experts。

重新归一化 retained masses。建议 pilot：

```yaml
bocpd:
  hazard: 0.01
  max_experts: 10
  restart_margin_rho_B: 1.0
  min_segment_length: 10
```

Hazard 需要做 matched / misspecified sensitivity，例如 $h/2,h,2h$。

若启用 `min_segment_length=L_min`，当当前 anchor age 小于 $L_{\min}$ 时令 fresh-branch hazard 为零；达到 $L_{\min}$ 后再恢复配置 hazard。该规则必须对所有 seeds 固定，并在结果中报告。

### 9.7 Expert assimilation

完成 BOCPD weight update、restart decision 和 pruning 后，对所有 retained expert assimilate 当前 observation：

- continuation experts 从缓存的 propagated state 开始做第 8.3 和 8.7 节更新；
- fresh expert 从 restart prior predictive state 开始更新；
- 每个 expert 独立维护 parameter particle weights 和 discrepancy posterior。

### 9.8 给 planner 的 posterior

BOCPD-BRPC 的总体 predictive law：

\[
p_t(Y_\star\mid X_\star)
=
\sum_{e\in\mathcal E_t}
\alpha_{e,t}
\sum_iw_{e,t}^{(i)}
\mathcal N
(Y_\star;\mu_{e,i},\Sigma_{e,i}).
\]

CE 使用该 mixture 的 predictive mean；PS 先采样 expert $e\sim\alpha_t$，再采样其 parameter/discrepancy state。

### 9.9 BOCPD-BRPC 单步伪代码

```text
BOCPD_BRPC_STEP(experts, anchor, X_t, Y_t):
    candidates = []

    for expert in experts:
        pred_cache = BRPC_PROPAGATE(expert.brpc_state)
        log_evidence = BRPC_LOG_PREDICTIVE(pred_cache, X_t, Y_t)
        new_log_mass = log(1-hazard) + expert.log_mass + log_evidence
        candidates.append((expert, pred_cache, new_log_mass))

    fresh = INIT_FROM_RESTART_PRIOR()
    fresh_cache = BRPC_PROPAGATE_OR_USE_PRIOR(fresh)
    fresh_log_evidence = BRPC_LOG_PREDICTIVE(fresh_cache, X_t, Y_t)
    candidates.append((fresh, fresh_cache,
                       log(hazard) + fresh_log_evidence))

    normalize_expert_log_masses(candidates)
    anchor = APPLY_HARD_ANCHOR_RULE(candidates, anchor, rho_B)
    candidates = PRUNE_TOP_EXPERTS(candidates, anchor, M_max)

    for expert, pred_cache, mass in candidates:
        expert.brpc_state = BRPC_ASSIMILATE_FROM_CACHE(pred_cache, X_t, Y_t)
        expert.log_mass = mass

    return candidates, anchor, restart_event
```

### 9.10 必须测试的 BOCPD 不变量

1. $\sum_e\alpha_{e,t}=1$；
2. 每个 expert 内 $\sum_iw_{e,t}^{(i)}=1$；
3. 当前 observation 在 pre-update evidence 之后恰好 assimilate 一次；
4. cache 保证 propagation 不重复采样；
5. 无 change 且 likelihood calibrated 时 restart frequency 与 hazard/threshold 相容；
6. noiseless synthetic jump 时 new expert posterior mass 应快速上升；
7. 当所有 action 下 old/new predictive law 相同时，detector 不应凭 action label 假检测。

---

## 10. CE Planner

CE 不考虑 hypothetical observation 对 future belief 的影响。

### 10.1 BRPC-CE model

当前 posterior predictive mean：

\[
\widehat f_t^{\mathrm{CE}}(s,a)
=
\sum_iw_t^{(i)}
\left[
f_{\mathrm{DT}}(s,a;\theta_t^{(i)})
+G(s,a)m_t^{(i)}
\right].
\]

### 10.2 BOCPD-BRPC-CE model

\[
\widehat f_t^{\mathrm{CE-CP}}(s,a)
=
\sum_e\alpha_{e,t}
\sum_iw_{e,t}^{(i)}
\left[
f_{\mathrm{DT}}(s,a;\theta_{e,t}^{(i)})
+G(s,a)m_{e,t}^{(i)}
\right].
\]

在 $M$-step rollout 中，belief 只做无 observation 的 drift/hazard prediction，不执行 Bayesian update。实现允许使用以下第一版近似：

- $\theta,m$ 按 transition mean 传播；
- BOCPD mass 按 hazard 做 prior prediction；
- 不根据模拟 next state 修改 weights。

CE 优化：

\[
a_{t:t+M-1}^{\mathrm{CE}}
=
\arg\max
\sum_{k=t}^{t+M-1}
\left[
\widehat r_k
-c_{\mathrm{sw}}(a_{k-1},a_k)
\right].
\]

只执行第一步，然后用真实 physical observation 更新 calibration 并重新规划。

---

## 11. Posterior-Sampling Planner

PS 在每个 physical decision time 采样一条 latent model trajectory，但 rollout 中不根据 hypothetical observations 更新 posterior。

### 11.1 Initial belief sample

BRPC：

1. $i\sim\operatorname{Categorical}(w_t)$；
2. $\theta_t=\theta_t^{(i)}$；
3. $u_t\sim\mathcal N(m_t^{(i)},C_t)$。

BOCPD-BRPC：

1. $e\sim\operatorname{Categorical}(\alpha_t)$；
2. $i\sim\operatorname{Categorical}(w_{e,t})$；
3. sample $\theta_t,u_t$ from that expert/particle。

### 11.2 Future latent path

在 horizon 内：

- 根据 $p(\theta_{k+1}\mid\theta_k)$ 传播参数；
- 根据 GP Markov prior 传播 $u_k$；
- BOCPD version 根据 hazard sample future change indicator；
- 若发生 simulated change，从 restart prior sample 新 $\theta,u$；
- 不使用模拟 observation 反向更新 sampled latent state。

### 11.3 Coherent discrepancy sample

先 sample inducing state $u$，然后使用

\[
\delta(x)=K_{xZ}K_{ZZ}^{-1}u
\]

作为一次 rollout 内 coherent discrepancy function。不要在每次 query 独立 sample 一个 marginal GP value，否则会把 epistemic uncertainty 错当成 white process noise。

### 11.4 PS variants

第一阶段主版本为 `PS-step`：每个 physical step 重新采样一个 latent trajectory 并重新规划。

`PS-commit`（采样后执行 $K>1$ 步）只作为后续 ablation，因为它会引入额外 commit-length 超参数。

---

## 12. Black-box action optimization 与公平 query budget

所有 CE、PS 和 oracle 使用同一个 CEM optimizer：

```yaml
cem:
  population: 512
  elite_fraction: 0.10
  iterations: 5
  smoothing: 0.20
  warm_start: true
```

Toy 可减少 population；CartPole 使用主设置。所有方法必须匹配：

- planning horizon；
- candidate action sequences 数；
- CEM iterations；
- simulator calls；
- process-noise treatment；
- action constraints；
- warm-start rule。

规划时默认使用 expected/noise-free transition；如果加入 Monte Carlo process noise，所有方法使用相同样本数和 common random numbers。

记录：

```text
num_twin_queries_per_step
num_twin_queries_total
planning_wall_clock_per_step
peak_memory
```

---

## 13. Oracle 定义

### 13.1 主 oracle：Current-Dynamics Oracle

Oracle 在每个 $t$ 知道当前真实

\[
(\theta_t,\delta_t)
\]

和当前可观测 state，但不知道：

- future changepoint time；
- future parameter/discrepancy realization；
- future process noise。

它在 planning horizon 内冻结当前 dynamics，并使用与 baseline 相同的 CEM 和 query budget。下一 physical step 后重新获得新的 current true dynamics。

该 oracle 衡量没有 calibration/detection delay 时的 ceiling，是主文唯一必报 oracle。

### 13.2 附录 oracle：Future-Regime Oracle

只在两个 toy 上实现。它知道 horizon 内 future

\[
(\theta_{t:t+M},\delta_{t:t+M})
\]

和 change schedule，但不知道 future process noise。它可以因 switching cost 在 change 之前提前移动。

不要实现“知道 future state/process noise”的 clairvoyant oracle；其上界过强且难以解释。

---

## 14. 实验矩阵

### 14.1 每个环境的主场景

1. stationary：sanity check；
2. gradual drift；
3. one abrupt change；
4. mixed drift + abrupt change：第一阶段完成后再加。

### 14.2 第一阶段方法

| ID | Calibration | Planner | 必做 |
|---|---|---|---:|
| `ce_brpc` | BRPC | CE | ✓ |
| `ps_brpc` | BRPC | PS-step | ✓ |
| `ce_bbrpc` | BOCPD-BRPC | CE | ✓ |
| `ps_bbrpc` | BOCPD-BRPC | PS-step | ✓ |
| `oracle_current` | true current dynamics | CEM-MPC | ✓ |
| `oracle_future` | future regime path | CEM-MPC | toy only |

### 14.3 主要 sweeps

- horizon $M$；
- switching coefficient $\lambda_\Delta$；
- drift speed；
- jump magnitude；
- cold-start size；
- hazard matched/misspecified；
- exploitation/diagnostic observability gap；
- BRPC tempering $\eta_\theta,\eta_\delta$。

不要对每个组合做 full Cartesian product。先固定主设置，每次只改变一个结构参数；最后选择 2D phase diagram：

\[
(\text{observability gap},\lambda_\Delta)
\quad\text{或}\quad
(\text{drift/change magnitude},\lambda_\Delta).
\]

---

## 15. 指标与统计协议

### 15.1 Primary

\[
J_T
=
\sum_{t=0}^{T-1}
\left[
r_{\mathrm{task},t}
-c_{E,t}
-c_{\mathrm{sw},t}
\right]
+r_T^{\mathrm{term}}.
\]

报告 paired regret：

\[
\operatorname{Regret}_T^{\mathrm{current}}
=J_T^{\mathrm{oracle-current}}-J_T^{\mathrm{method}}.
\]

### 15.2 Secondary

- task reward、energy cost、switching cost、failure cost 分项；
- parameter RMSE；
- physical next-state/response RMSE；
- negative log predictive density；
- credible interval coverage；
- BOCPD restart precision/recall/delay；
- post-change time-to-recover；
- action total variation；
- number of switches / large relocations；
- simulator calls、wall-clock 和 memory。

### 15.3 Detection metrics

若真实 changepoint 为 $\tau_j$，预测 restart 为 $\widehat\tau_k$，使用 tolerance window $\pm d$：

- event-level precision@$d$；
- recall@$d$；
- detection delay $\widehat\tau-\tau$；
- false restarts per 1000 steps。

主设置可用 $d=2$（toy）和 $d=10$（CartPole），但同时画 cumulative expert mass：

\[
p_t^{\mathrm{recent}}
=
\sum_{e:t-s_e\le d}\alpha_{e,t}.
\]

### 15.4 重复与置信区间

- Toy：至少 50 paired deployment seeds；
- CartPole：至少 20 paired seeds，资源允许时 30；
- 报 mean、median、standard error 和 95% paired bootstrap CI；
- 同时报告 90% cost/return quantile 作为 tail diagnostic，但不改变 primary objective。

---

## 16. 软件接口建议

### 16.1 Environment

```python
class EvolvingPhysicalEnv:
    def reset(seed, latent_path=None) -> Observation
    def step(action) -> tuple[Observation, RewardBreakdown, Done, Info]
    def get_true_latent_for_oracle() -> LatentState

class DigitalTwin:
    def step(state, action, theta) -> next_state_or_response
```

`RewardBreakdown` 必须包含：

```text
task_reward
energy_cost
switching_cost
failure_cost
net_reward
```

### 16.2 Calibrator

```python
class Calibrator:
    def initialize(cold_start_dataset)
    def predict(inputs) -> PredictiveMixture
    def update(inputs, outputs)
    def sample_latent(rng) -> LatentSample
    def predictive_mean_model() -> ModelHandle
    def diagnostics() -> dict
```

BOCPD-BRPC 另外返回：

```text
expert_masses
expert_start_times
anchor_start_time
restart_event
recent_change_probability
```

### 16.3 Planner

```python
class RecedingHorizonPlanner:
    def act(state, previous_action, calibrator, reference_schedule) -> action
```

### 16.4 完整 deployment loop

```text
initialize env with paired latent/noise path
initialize calibrator from shared cold-start data
initialize previous_action

for t in 0 ... T-1:
    action = planner.act(state, previous_action, calibrator, schedule)
    next_state, reward_parts = physical_env.step(action)

    X_t = concat(state, action)
    Y_t = next_state
    calibrator.update(X_t, Y_t)

    log reward parts, posterior diagnostics, detector diagnostics,
        action, state, true latent (evaluation only), query count, timing

    state = next_state
    previous_action = action
```

Toy 2 将 $Y_t$ 替换为 observed physical response。

---

## 17. 数值稳定性与单元测试

### 17.1 数值规则

- 所有 mixture/expert/particle weights 使用 log domain；
- covariance 添加 adaptive jitter；
- 所有 matrix inverse 替换为 Cholesky solve；
- covariance 每步对称化：

\[
C\leftarrow(C+C^\top)/2;
\]

- 若最小 eigenvalue 小于 tolerance，增加 jitter 并记录；
- state/action/kernel inputs 标准化；
- CartPole 不同 output dimensions 标准化后再形成 likelihood；
- 固定所有 random-number stream 并分别记录 environment、calibration、planner seeds。

### 17.2 必做 unit tests

1. **No discrepancy, static parameter：** BRPC particle posterior 收敛到真值；
2. **Known linear-Gaussian case：** discrepancy information update 与直接 Gaussian conditioning 一致；
3. **Resampling coupling：** $\theta^{(i)}$ 与 $m^{(i)}$ index 始终对应；
4. **Predictive normalization：** particle/expert mixture weights 各自和为 1；
5. **No-change BOCPD：** false restart 可控；
6. **Large jump BOCPD：** fresh expert mass 在有限步内超过 anchor；
7. **Prequential ordering：** 当前 observation 不得先 update 再用于 detector evidence；
8. **CE determinism：** 固定 belief/seed 时 action 可复现；
9. **PS coherence：** 同一 inducing sample 在 rollout 中形成 coherent function；
10. **Reward accounting：** net reward 等于各分项代数和；
11. **Oracle isolation：** 非 oracle 代码路径无法访问 true latent；
12. **Paired paths：** 不同方法在相同 evaluation seed 下获得相同 latent/noise schedule。

---

## 18. 分阶段交付顺序

### Milestone A：环境与 geometry

- 实现三个 environment；
- 实现 frozen latent/noise path；
- 输出第 3 节四类 geometry plots；
- 验证 old/new optimal action 和 oracle gap。

没有通过 geometry gate，不进入下一阶段。

### Milestone B：BRPC calibration only

- fixed-support BRPC-F；
- cold-start initialization；
- static/drift/jump calibration curves；
- unit tests 1-4。

### Milestone C：BOCPD-BRPC

- expert mixture、restart prior、prequential evidence；
- hard-anchor restart 和 pruning；
- calibration/detection metrics；
- unit tests 5-7。

### Milestone D：CE、PS、oracle

- common CEM optimizer；
- four baseline combinations；
- current oracle；
- toy future oracle；
- query-budget matching。

### Milestone E：正式实验

- stationary sanity；
- gradual drift；
- abrupt change；
- horizon/switching/observability sweeps；
- paired confidence intervals 和 phase diagrams。

---

## 19. Go/No-Go 判据

在开发 proposed method 前，至少满足：

1. stationary matched case 中 BRPC calibration 与 oracle/ground truth 一致，说明实现正确；
2. gradual drift 中 BRPC 明显优于错误的 static/no-drift filter（仅作为 calibration sanity）；
3. abrupt change 中 BOCPD-BRPC 明显缩短 calibration recovery；
4. 即使使用 BOCPD-BRPC，CE/PS 与 current oracle 之间仍有显著累计-return gap；
5. 该 gap 随 observability gap、switching cost 和 planning horizon 呈结构性变化；
6. gap 不能仅由增加 CEM budget 消除；
7. Toy 2 中 diagnostic point 对任何 sampled single model 都不是生产最优，但对 Bayes multi-step control 有潜在价值；
8. CartPole 中结论至少在两种 cold-start size 和两种 change magnitude 下保持。

若第 4 条不成立，则 BOCPD calibration 已经解决主要困难，不应继续声称需要新的 dual planner。若只有极端超参数下成立，则重新设计 geometry，而不是挑选单个有利 seed。

---

## 20. 第一阶段明确不做的扩展

以下内容留到 baseline gap 被确认之后：

- BOCPD belief 直接进入 proposed planning operator；
- CUSUM-BRPC baseline；
- myopic changepoint information-gain baseline；
- KH / scenario-tree dual control；
- evolving full GP hyperparameters；
- joint multi-output discrepancy GP；
- real hardware CartPole；
- Plant Simulation / proprietary simulator；
- safety-constrained or risk-sensitive objective。

该顺序保证下一步方法是由已经观察到的 baseline failure mode 推导出来，而不是先设计方法、再反向挑 benchmark。

---

## 21. 实现时应对照的来源

1. Yang Xu and Chiwoo Park, *Online Bayesian Calibration under Gradual and Abrupt System Changes*：BRPC projected particle update、fixed-support discrepancy update、B-BRPC expert recursion与 restart rule。Local source: `/Users/bytedance/Downloads/Online_Bayesian_Calibration.pdf`；extended version: <https://arxiv.org/abs/2605.06612>。
2. Ryan Prescott Adams and David J. C. MacKay, *Bayesian Online Changepoint Detection*：run-length posterior 与 prequential recursion。<https://arxiv.org/abs/0710.3742>。
3. R. Alami, O. Maillard, and R. Féraud, *Restarted Bayesian Online Change-Point Detector Achieves Optimal Detection Delay*：hard-restart BOCPD 原则。
4. M. Wabersich and M. Zeilinger, *Bayesian Model Predictive Control: Efficient Model Exploration and Regret Bounds Using Posterior Sampling*：PS-MPC baseline 定义。<https://proceedings.mlr.press/v120/wabersich20a.html>。
5. SensorsINI physical CartPole repository：physical interface、calibration 与 nominal simulator 实现参考。<https://github.com/SensorsINI/physical-cartpole>。

若本文档与 BRPC 原论文公式冲突，以论文公式为准，并在实现日志中记录差异；benchmark、planner、oracle 和公平比较协议以本文档为准。
