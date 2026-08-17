# Evolving Digital Twin: 两个首轮 benchmark 的设计与实验协议

> 状态：v0.3，供讨论与首轮实现  
> 核心目标：先验证问题是否真实存在、现有 dual-control 方法在什么条件下失效，再决定我们的方法。本文档暂不预设 proposed method。首轮不实现 BRPC/proposed planner，而是把 benchmark 作为 falsification protocol：如果调好的现有 dual-control baselines 已经接近 Bayes reference，就及时停止该问题方向。

## 1. 我们究竟要测什么

我们研究的是一条**不可重放的 continuous-deployment trajectory**。每个 physical action 同时产生：

1. 当期真实 operational reward/cost；
2. 一条新的 physical transition，用于更新 evolving digital twin；
3. 对后续状态、后续可辨识性和未来决策的影响。

因此主要评价量不是“最后学到的 policy 有多好”，也不是“最终参数估计误差”，而是部署期间已经真实发生的累计回报：

\[
J_T^\pi = \sum_{t=0}^{T-1} r_t+r_T^{\mathrm{term}},
\qquad
C_T^\pi=-J_T^\pi.
\]

Primary metric 使用 undiscounted realized return（即 \(\gamma=1\)），因为早期探索和迁移造成的 physical cost 不能通过 discount 被弱化；折扣只可作为 planning-horizon sensitivity analysis。

一条 deployment trajectory 内不 reset、不重放，也不允许把早期探索代价从结果里删掉。为估计方差，可以用不同随机种子重复完整 deployment；这只是统计重复，不是算法可以利用的 episode reset。

### 1.1 要检验的四个假设

- **H1：静态 sanity check。** 当参数确实静态、twin 正确指定时，Klenske-Hennig approximate dual control（下文记为 KH-AD）应当接近相应 oracle；否则说明复现或实现有问题。
- **H2：慢漂移。** 静态参数 posterior 会逐渐过度自信，导致旧数据权重过大、重新校准太慢，从而产生持续 tracking regret。
- **H3：突变后的 posterior lock-in。** 在很长的稳定阶段之后发生 change point，静态 KH-AD 可能非常确信旧参数，短规划窗内看不到重新辨识的长期价值，因此产生明显的 post-change regret。
- **H4：显式 model discrepancy。** 即使继续 probing，若 physical dynamics 不属于 simulator family，参数 posterior 也可能“很窄但很错”；只在 \(\theta\) 上做 dual control 不能消除结构性偏差。

其中 H2/H3 是第一轮核心；H4 是 benchmark 1 的第二阶段扩展。若一个调好过程噪声的 dynamic KH baseline 已经在合理的 drift/jump 下接近 Bayes oracle，那么“遗忘/非平稳性”本身不足以支撑论文，需要把 novelty 放到 discrepancy、change adaptation 或 simulator cost 上。

## 2. 所有 benchmark 共用的协议

### 2.1 Digital twin 与 physical system 的角色

- **Digital twin**：算法内部可调用的 simulator/world model；输入状态、动作和 calibration parameter，输出下一状态或 response。它可以被多次调用，但 simulator call 数和 wall-clock time 要记录。
- **Physical system**：只沿当前真实轨迹前进一步；其 transition 不可查询后撤，也不可为同一状态尝试多个动作。
- **Physical data**：不是额外免费获得的数据集。每一步只有被执行动作产生的 \((s_t,a_t,r_t,s_{t+1})\)，而该动作的 operational cost 已计入累计代价。首轮还应单独测试 physical data 稀疏的情形：physical sampling 可能远慢于 digital-twin rollout，因而每条 deployment 中可用于 recalibration 的真实 transitions 很少；这正是 digital twin 相比纯 model-free online learning 的重要动机之一。
- **参数**：\(\theta_t\) 是 physical system 的隐变量。控制器看不到它，只能通过 physical transitions 推断。首轮先假设 \(\theta_t\) 的演化是 exogenous，即 \(p(\theta_{t+1}\mid\theta_t)\)，不让 action 直接影响 degradation/wear；action-dependent \(p(\theta_{t+1}\mid\theta_t,s_t,a_t)\) 只作为后续扩展。

### 2.2 统一的 physical reward 与 action cost

两个 benchmark 都把算法目标写成最大化真实部署期间的净 reward：

\[
r_t=
r_{\mathrm{task}}(s_t,a_t)
-\lambda_E c_E(a_t)
-\lambda_\Delta c_\Delta(a_t,a_{t-1})
-M\mathbf 1\{\mathrm{failure}\}.
\]

其中 \(r_{\mathrm{task}}=-\ell_{\mathrm{task}}\)，即 tracking/stabilization error 越小，task reward 越高。

第一轮先使用 quadratic switching cost 作为可导 sanity setting；之后单独加入 non-smooth migration cost，例如 \(k\mathbf 1\{|a_t-a_{t-1}|>\epsilon\}\) 或离散 recipe 下的 \(k\mathbf 1\{a_t\ne a_{t-1}\}\)，测试现有 dual-control / MPC 近似是否依赖平滑 cost landscape。

等价地，代码中最小化 stage cost \(\ell_t=-r_t\)。这里要区分两类 action cost：

- \(c_E(a_t)=\|a_t\|_2^2\)：持续施加动作所需的 control effort / energy；
- \(c_\Delta(a_t,a_{t-1})=\|a_t-a_{t-1}\|_2^2\)：切换 recipe、快速 ramping、重新配置或 transient risk，即本文重点讨论的 migration cost。

因此，如果用统一记号 \(c(a_t,a_{t-1})\)，它应表示

\[
c(a_t,a_{t-1})
=\lambda_E\|a_t\|_2^2
+\lambda_\Delta\|a_t-a_{t-1}\|_2^2,
\]

而不应把 energy 与 switching cost 混成同一个物理概念。最终评价量为

\[
C_T^\pi=\sum_{t=0}^{T-1}\ell_t+\ell_T^{\mathrm{term}},
\qquad
J_T^\pi=-C_T^\pi.
\]

reward/cost 中**不显式加入** posterior variance、information gain、UCB/LCB bonus 或 calibration error。信息价值应当通过

\[
a_t\longrightarrow s_{t+1}\longrightarrow b_{t+1}
\longrightarrow\text{future physical reward}
\]

自然进入 belief-space planning。这样最终比较的始终是实际 physical trajectory 上已发生的累计 reward，而不是人为设计的信息分数。

### 2.3 两类参数演变

所有方法在完全相同的 \(\theta_{0:T}\)、过程噪声和观测噪声轨迹上配对比较。首轮默认 \(\theta_t\) 演化不依赖 action；这有助于把“系统本身演化”与“action-induced wear/degradation”分开归因。

**Slow drift：有界 OU / random-walk process**

\[
\theta_{t+1}=\Pi_{\Theta}\!\left[
\bar\theta+\rho(\theta_t-\bar\theta)+\sigma_\theta\varepsilon_t
\right],
\qquad \varepsilon_t\sim\mathcal N(0,1).
\]

建议先用 \(\rho\in\{0.995,0.999\}\)，再用 pilot 调 \(\sigma_\theta\) 得到 mild / medium / hard 三档。OU 比固定斜率更适合作为正式结果，因为算法不能直接背下一个确定的时间函数；线性漂移或正弦漂移只作为可视化 sanity check。

**Piecewise constant：固定 change point + 随机 hazard 两层实验**

第一层使用固定 change points，便于观察 posterior lock-in 和 recovery curve：

\[
\theta_t=
\begin{cases}
\theta^{(0)}, & t<\tau_1,\\
\theta^{(1)}, & \tau_1\le t<\tau_2,\\
\theta^{(2)}, & t\ge \tau_2.
\end{cases}
\]

第二层再使用不可预测的 hazard model：每一步以概率 \(h\) 进入新 regime，并设置最短 regime 长度。固定 change point 是诊断工具，随机 change point 是正式 robustness test。

### 2.4 公平比较所需的两个 oracle

“Oracle”必须拆成两个，不能混用：

1. **Bayes oracle（主要公平参照）**：知道真实的 \(p(\theta_{t+1}\mid\theta_t)\)、噪声模型和 physical transition family，但看不到 realized \(\theta_t\)。它仍然只能利用截至当前的 physical history，并在 belief space 中规划。
2. **Clairvoyant oracle（性能上界）**：每一步直接观察 realized \(\theta_t\)，再进行 MPC/control。它比任何实际方法拥有更多信息，因此只用于给出 ceiling，不作为“我们必须击败”的 baseline。

报告两个 gap：

\[
\mathrm{Regret}_{\mathrm{Bayes}}=C_T^\pi-C_T^{\mathrm{Bayes}},
\qquad
\mathrm{Gap}_{\mathrm{clair}}=C_T^\pi-C_T^{\mathrm{clair}}.
\]

Toy benchmark 中 Bayes oracle 可通过参数网格、Gaussian quadrature 和 belief-state dynamic programming 做成 numerical near-exact oracle。CartPole 中精确 Bayes oracle 不现实，使用高预算 particle-belief MPC / scenario-tree planner，并明确称作 **high-budget Bayes reference**，避免声称“exact oracle”。

### 2.5 第一轮 baseline

| 方法 | 参数 belief | 是否考虑 action 的信息价值 | 用途 |
|---|---|---:|---|
| Clairvoyant MPC | 直接看到 \(\theta_t\) | 不需要 | 不可达到的性能上界 |
| Bayes oracle/reference | 正确 dynamic model | 是 | 公平的主要 oracle |
| KH-AD-static | 假设 \(\theta\) 永远不变 | 是，KH 的 approximate dual term | 被检验的原始方法 |
| KH-AD-RW | \(\theta_{t+1}=\theta_t+\omega_t\) | 是 | 强 baseline；过程噪声在 validation seeds 上调参 |
| CE-static | 静态 posterior mean | 否 | 分离“dual effect”与纯估计作用 |
| CE-RW | dynamic posterior mean | 否 | 分离“tracking/forgetting”与 planning 作用 |

可在第二轮加入 Thompson MPC、myopic UCB/LCB、robust/risk-sensitive MPC；第一轮不必加入 model-free/meta-RL，以免掩盖我们真正要诊断的机制。

### 2.6 稀疏 physical data / fast digital twin 测试

除每个 control step 都产生 physical transition 的标准设置外，首轮应加入一个单独的 sparse-physical-data stress test。动机是 digital twin 场景下 physical sampling 往往慢、贵或有安全约束，而 twin rollout 可以快很多；因此算法可能每个 physical interval 内能做大量 simulator planning，但只能偶尔获得真实 recalibration signal。

建议先实现两种等价诊断方式：

1. **Subsampled observation / update**：physical system 每一步仍真实前进并计入 cost，但只有每 \(m\) 步获得一次用于 posterior update 的 transition summary，\(m\in\{1,5,10,20\}\)。未观测步只用于累计 reward/cost，不能被算法当成免费 calibration data。
2. **Coarse physical decision interval**：一个 physical action/recipe 持续 \(m\) 个 simulator time steps，期间 digital twin 可用于内部 rollout，但真实系统只在 interval 末端回传一个 aggregated transition/reward。这个版本更贴近 physical sampling frequency 低于 digital-twin simulation frequency 的 setting。

在 sparse setting 下，\(\theta_t\) 或 \(\delta_t\) 的 drift 应相对 physical sampling 更慢，或者以 sudden change 形式出现；否则所有方法都会因为信息论限制同时失败，无法诊断 dual-control 方法本身。报告时需要同时给出 physical transitions 数、可用于 calibration 的 observed transitions 数、digital-twin calls 数和 wall-clock time。

### 2.7 统一评价指标

**Primary metric**

- physical cumulative cost \(C_T\)（或等价 cumulative reward \(-C_T\)）；
- 对 Bayes oracle 的 paired regret；
- 每个 regime 内以及 change point 后的 cumulative regret。

**Secondary metrics**

- immediate operational cost、control-energy cost、switching cost、failure cost 分项；
- change 后 time-to-recover：rolling cost 回到 oracle 的 10% 容差带所需步数；
- \(\theta_t\) posterior mean error、negative log predictive density、credible-interval coverage；
- posterior entropy/variance 与真实 calibration error 的关系，用于识别“自信但错误”；
- physical transitions 数、observed calibration transitions 数、digital-twin calls 数和 wall-clock time。

所有曲线画 paired mean/median 和 95% bootstrap CI；同时报告 tail risk（90% 或 95% quantile cost），但 primary objective 仍是 realized cumulative cost。这里的 quantile 是**评价指标**，不先偷换成方法的优化目标。

---

## 3. Benchmark A：Evolving-Actuator CartPole Digital Twin

### 3.1 为什么推荐这个 benchmark

CartPole 有三个优点：

1. 是 ICLR/ML audience 熟悉的连续控制环境，图像和 failure mechanism 直观；
2. simulator 可解析且 rollout 快，便于做大量消融和高预算 Bayes reference；
3. 将 calibration parameter 设为 actuator effectiveness 后，动作大小直接决定信息量，形成清晰的 dual effect：小动作安全但难辨识，大动作更有信息却付出能量、切换和失稳风险。

这不是把 Gym CartPole 当成“真实硬件”；它是可复现的 **simulated physical benchmark**。最终论文若有条件，可再补一个真实系统或公开 industrial dataset，但首轮无需等待硬件。

### 3.2 状态、动作和任务

状态和动作定义为

\[
s_t=(p_t,\dot p_t,\phi_t,\dot\phi_t),
\qquad u_t\in[-u_{\max},u_{\max}],
\]

其中 \(p\) 是 cart position，\(\phi=0\) 表示 pole upright，\(u\) 是 commanded force。执行器实际施加的力为

\[
F_t=\theta_t u_t,
\qquad \theta_t\in[0.55,1.45].
\]

核心任务是 pole stabilization + 已知 reference schedule 下的 cart-position tracking。将目标位置记为 \(p_t^{\mathrm{ref}}\)，建议每 100-150 步在 \(\{0,+0.8,0,-0.8\}\) 间平滑切换，使 actuator uncertainty 确实会改变未来最优动作，而又不需要人为加入“信息奖励”。

### 3.3 Digital twin

Digital twin 使用标准离散化 nonlinear CartPole equations：

\[
\tilde s_{t+1}=f_{\mathrm{CP}}(s_t,\theta_tu_t;\psi_{\mathrm{nom}})+\epsilon_t.
\]

其中几何参数、质量和重力组成已知 nominal physics \(\psi_{\mathrm{nom}}\)，唯一在线 calibration parameter 是 actuator gain \(\theta_t\)。首轮先保持一维参数，避免结果只是“高维模型没学好”。posterior update 可用 EKF/UKF、particle filter，或在 transition residual 上做一维 Bayesian regression；所有 controller 必须共享同一个 filter，除非该 filter 本身就是被消融的对象。

### 3.4 Physical system：两级难度

**A0 - matched family（首轮主实验）**

\[
s_{t+1}=f_{\mathrm{CP}}(s_t,\theta_tu_t;\psi_{\mathrm{nom}})+\xi_t.
\]

它与 twin family 相同，只有 \(\theta_t\) 演变。这个版本纯粹检验 static posterior、dynamic filtering 和 dual planning，不能用 model misspecification 为失败找借口。

**A1 - misspecified physical system（第二阶段）**

physical system 再加入 twin 没有的 actuator lag 和 Coulomb friction：

\[
F_{t+1}=(1-\alpha)F_t+\alpha\theta_tu_t,
\qquad
s_{t+1}=f_{\mathrm{CP}}(s_t,F_t;\psi_{\mathrm{nom}})
+d_c(s_t;f_c)+\xi_t,
\]

其中 \(d_c\) 是由 Coulomb friction 产生、作用于 cart acceleration 分量的修正项。算法的 basic twin 仍只含 \(\theta_tu_t\)。这样可以单独测试 structured calibration 无法解释的 \(\delta(s,a)\)。A1 不应与 A0 混成一个实验，否则无法判断 gap 来自 nonstationarity 还是 discrepancy。

### 3.5 Physical reward/cost

每一步定义

\[
\ell_t=
w_p(p_t-p_t^{\mathrm{ref}})^2+w_\phi\phi_t^2
+w_v\dot p_t^2+w_\omega\dot\phi_t^2
+\lambda_Eu_t^2+\lambda_\Delta(u_t-u_{t-1})^2
+M\mathbf 1\{\text{failure}\},
\qquad r_t^{\mathrm{env}}=-\ell_t.
\]

其中前四项是 task-tracking/stabilization cost，\(\lambda_Eu_t^2\) 是 energy/control-effort cost，\(\lambda_\Delta(u_t-u_{t-1})^2\) 是 migration/ramping cost。这个 reward 不奖励“知道得更多”；只有当新信息改善未来 physical control 时，它才间接有价值。

初始建议值（只作为 pilot 起点）为

\[
(w_p,w_\phi,w_v,w_\omega,\lambda_E,\lambda_\Delta)
=(1,20,0.1,0.1,10^{-3},10^{-2}).
\]

部署结束时再计一次 terminal cost：

\[
\ell_T^{\mathrm{term}}
=w_{p,T}(p_T-p_T^{\mathrm{ref}})^2+w_{\phi,T}\phi_T^2.
\]

failure 定义为 \(|p_t|>2.4\) 或 \(|\phi_t|>20^\circ\)。failure 后进入 absorbing state，并对剩余 horizon 计固定 failure cost；不允许通过 reset 抹掉失败。\(M\) 应校准到“失败显著昂贵但不完全吞没所有正常成本”。

### 3.6 参数演变和 horizon

- deployment horizon：首轮 \(T=600\)，控制间隔 \(\Delta t=0.02\) 或 \(0.05\) 秒；
- static：\(\theta_t\equiv1.0\)；
- slow drift：\(\bar\theta=1.0,\rho=0.995\)，从 \(\sigma_\theta\in\{0.002,0.005,0.01\}\) pilot；
- deterministic piecewise：\(1.0\rightarrow0.65\rightarrow1.25\)，change points \(\tau_1=200,\tau_2=400\)；
- random piecewise：平均 regime 长度约 150-250 步，并设置至少 80 步的 minimum dwell time。

上述幅度必须经过 pilot：我们希望 medium 难度下 clairvoyant failure rate 接近 0，CE-static/KH-static 不至于 100% failure，同时出现可测的 recovery gap。

### 3.7 这个 benchmark 的关键图

1. 单条代表性 deployment：\(\theta_t\)、posterior mean/interval、action、state cost、累计 regret 五行对齐；
2. change point 对齐后的 average recovery curve；
3. 总 physical cost 与 switching/failure 分解；
4. posterior confidence vs. true error scatter，用来显示是否存在 confident-but-wrong；
5. performance vs. twin-call budget。

---

## 4. Benchmark B：Klenske-Hennig Scalar Dual-Control Toy 的动态扩展

### 4.1 先原样复现 paper 的静态两步系统

Klenske and Hennig (JMLR 2016) 的 scalar system 为

\[
x_{k+1}=a x_k+b u_k+\xi_k.
\]

Section 6.1 使用的设置是：\(a=1\) 已知，真实 \(b=2\)，先验 \(p(b)=\mathcal N(1,10)\)，过程噪声 \(Q=10^{-1}\)，观测噪声 \(R=0\)，状态权重 \(W=1\)，动作权重 \(\Lambda=1\)，规划 horizon \(T=2\)。paper 在这个极小问题上比较了 approximate dual 与近似 exact sampling solution。

**第一项验收测试**：在扩展到 dynamic \(b_t\) 之前，必须复现 paper 中“KH-AD 的 cost landscape 明显比 CE 更接近 exact dual”的定性结果。若这个测试不通过，不继续解释动态实验。

原论文链接：[Dual Control for Approximate Bayesian Reinforcement Learning](https://www.jmlr.org/papers/v17/15-162.html)。

### 4.2 Continuous-deployment extension

将固定 \(b\) 改为演变的 calibration state：

\[
x_{t+1}=x_t+b_tu_t+\xi_t,
\qquad y_{t+1}=x_{t+1}+\eta_t.
\]

为了保留 paper reproduction，主设置先取 \(R=0\)；之后再加小观测噪声作为 robustness test。动作限制为 \(u_t\in[-3,3]\)，防止 numerical oracle 选择无限大的探测动作。

Digital twin 为

\[
\tilde x_{t+1}=x_t+\tilde b\,u_t+\tilde\xi_t.
\]

KH-AD-static 在每个 receding-horizon window 内以及跨时间都把 \(b\) 当成同一个未知常数。每次只执行规划得到的第一个动作，然后用唯一发生的 physical transition 更新 posterior。

观测提供的 calibration signal 是

\[
z_t=x_{t+1}-x_t=b_tu_t+\xi_t.
\]

因此 \(|u_t|\) 越大，通常关于 \(b_t\) 的 Fisher information 越大，但动作代价也越高；\(u_t=0\) 几乎不给出参数信息。这使 dual effect 完全透明，也方便计算 near-exact Bayes oracle。

### 4.3 Physical cost

动态扩展的 target 是 \(x_t^{\mathrm{ref}}=0\)，任务是持续把受过程扰动的状态稳定在零点。为避免 \(x_0=0,u_t=0\) 的平凡初始化，首轮固定 \(x_0=1\)；robustness test 再使用 \(x_0\sim\mathcal N(1,0.1^2)\)。每一步使用

\[
\ell_t=
(x_t-x_t^{\mathrm{ref}})^2
+\lambda_E u_t^2
+\lambda_\Delta(u_t-u_{t-1})^2,
\qquad r_t=-\ell_t.
\]

部署结束时使用 terminal cost

\[
\ell_T^{\mathrm{term}}
=w_T(x_T-x_T^{\mathrm{ref}})^2.
\]

- paper reproduction：\(\lambda_E=1,\lambda_\Delta=0,w_T=1,T=2\)；
- continuous deployment：建议 \(\lambda_E=0.1,w_T=1\) 起步，\(\lambda_\Delta\in\{0,0.05\}\)，\(T_{\mathrm{dep}}=300\)，planning horizon \(H\in\{2,5,10\}\)。

原论文的有限时域 cost 包括每一步 state-tracking cost、control cost，以及最后的 terminal-state cost。复现阶段严格保留这一形式。加入 \(\lambda_\Delta\) 后，旧参数下的控制动作、重新 probing 和 regime migration 都会产生真实成本；但必须同时保留 \(\lambda_\Delta=0\) 版本，避免所有结果仅由我们新加的 switching penalty 驱动。

### 4.4 两种 \(b_t\) 演变

**Slow drift**

\[
b_{t+1}=\Pi_{[0.4,2.6]}
\left[2+\rho(b_t-2)+\sigma_b\varepsilon_t\right],
\quad \rho=0.995.
\]

从 \(\sigma_b\in\{0.005,0.015,0.03\}\) pilot。正式结果不让 baseline 知道 realized noise；Bayes oracle 知道正确的 transition law。

**Piecewise constant**

- mild：\(2.0\rightarrow1.2\rightarrow2.2\)；
- hard：\(2.0\rightarrow0.6\rightarrow1.8\)；
- change points：\(t=100,200\)。

第一轮不建议直接 sign flip。\(b:2\rightarrow-1.5\) 很醒目，但可能把任务变成极端 instability test，使任何遗忘机制都显得必要；它可以作为 appendix stress test，而不是主结果。

### 4.5 Toy 中 oracle 的具体实现

由于状态、动作和参数都是一维，可以把 belief state 表示为 parameter grid 上的 posterior weights：

1. 在 \(b\in[0.3,2.7]\) 和相关的 \(x\) 范围上建细网格；
2. 根据正确的 drift/jump kernel 做 prediction；
3. 对 hypothetical \(z_t\) 用 Gaussian quadrature 更新 posterior；
4. 对动作网格做 finite-horizon Bellman backup；
5. 逐步加密 \(x\)、\(b\)、\(u\) 和 observation grids，直到 root action 与 value 基本收敛。

这得到的是 **numerical near-exact Bayes oracle**。同时计算 clairvoyant controller：它直接使用真实 \(b_t\)。数值收敛误差应单独报告，而不是将有限网格结果直接叫作 exact oracle。

### 4.6 Toy 的关键诊断

- 静态 \(b\)：KH-AD 应接近 Bayes oracle；
- slow drift：比较 KH-AD-static 与 KH-AD-RW 的 steady-state tracking regret；
- long stable period + jump：比较 change 前 posterior variance、change 后 bias、重新产生 informative action 的延迟；
- 将 KH-AD-RW 的 process noise 从过小扫到过大，展示 stability-plasticity trade-off：遗忘太少会 lock in，遗忘太多会长期保持高方差并过度 probing；
- 扫 planning horizon \(H\)，检查失败究竟来自错误 evolution model，还是仅仅短视。

---

## 5. 首轮实验矩阵

建议按以下顺序执行，不要一开始就跑完整大表：

| 阶段 | 环境 | regime | 目的 | 通过条件 |
|---|---|---|---|---|
| B0 | Scalar | static, \(T=2\) | 复现 KH paper | KH-AD cost landscape 接近 paper/exact reference |
| B1 | Scalar | long static | 检查 receding-horizon 实现 | KH-AD-static 不应系统性输给 CE-static |
| B2 | Scalar | slow drift | 测 steady tracking | gap 随 drift severity 有规律变化 |
| B3 | Scalar | fixed jumps | 测 posterior lock-in/recovery | 能定位 gap 出现在哪个 change 后区间 |
| B4 | Scalar | random jumps | 检查非脚本化 robustness | 结论在随机 change time 下保持 |
| B5 | Scalar | sparse physical data | 测低采样频率 / 少量真实数据 | gap 不只是由无限 physical data 假设消失 |
| B6 | Scalar | non-smooth switch cost / multimodal regimes | 测不可导 migration cost 与 posterior 多峰 | 定位 smooth/Gaussian 近似是否失效 |
| A0 | CartPole matched | static | controller/filter sanity | oracle 和强 baseline 基本稳定 |
| A1 | CartPole matched | slow drift | 测 dynamic calibration + control | 非平稳 gap 可重复且不靠 failure 饱和 |
| A2 | CartPole matched | jumps | 测恢复与真实累计损失 | post-change regret 与 recovery 可解释 |
| A3 | CartPole misspecified | drift/jumps | 测 discrepancy | 识别 confident-but-wrong，而非仅增大噪声 |

### 5.1 随机种子和统计量

- Scalar：建议至少 1,000 条 paired deployment trajectories；计算便宜，可做到 5,000。sparse physical data 版本需要扫 observation interval \(m\)，但仍使用 paired seeds。
- CartPole：先用 100 条 pilot，正式结果至少 200 条 paired trajectories。
- 相同 seed 下共享 \(\theta\) path、process noise、observation noise、physical observation mask 和初始状态。
- 超参数只在独立 validation seeds 上调；test seeds 不再调 process noise、planning horizon 或 cost weights。
- 报告 paired bootstrap 95% CI 和 standardized effect size，不能只给单条漂亮轨迹。

### 5.2 计算预算

分两层报告：

1. **Correctness layer**：给各方法足够 simulator calls，先判断统计/决策假设本身是否有 gap；
2. **Budget layer**：将每一步 twin calls 限制在相同预算，例如 \(10^2,10^3,10^4\)，比较 performance-computation Pareto curve。

如果 computer model 后来替换为昂贵 simulator，可以在 benchmark A 上给 twin 加人工 latency，或训练 emulator；但不建议第一轮同时引入 surrogate approximation，否则无法区分性能差来自规划、calibration 还是 surrogate error。

## 6. Pilot 校准规则：避免“为我们的方法造题”

参数不是为了让某个方法赢，而是要形成可辨识的难度梯度。建议预先写下以下校准标准：

- static matched setting 中，KH-AD 应与 Bayes reference 足够接近；
- medium drift 下，KH-AD-static 相对 Bayes oracle 出现可测但非灾难性的 gap，例如总 cost 高 10%-30%；
- fixed jump 后应出现约 10-50 步的可见 recovery interval，而非 1 步恢复或永远崩溃；
- clairvoyant failure rate 低，所有非 oracle 方法的 failure rate 不能同时接近 100%；
- 结果应覆盖 mild/medium/hard，而不能只汇报一个最有利的 jump size；
- 若 gap 只在 sign flip、极大 drift、不现实噪声或极端稀疏到所有方法都无法辨识的 physical data 下出现，则 benchmark 不足以支持主要 claim。

这些百分比只是 pilot target，不是要在最终实验中筛选“会赢的 seeds”。一旦用 validation set 定下参数，test set 必须冻结。

## 7. 什么结果会否定我们的当前叙事

以下结果都应被当成有价值的 falsification，而不是隐藏：

1. KH-AD-RW 在合理调参后，在 slow drift 和 jumps 上都接近 Bayes oracle：说明动态 prior 已足够，单讲 evolving \(\theta\) 不新；
2. KH-AD-static 在 jump 后会立即主动 probing 并快速恢复：说明 posterior lock-in 并没有形成主要 gap；
3. CartPole 的 gap 完全来自 failure penalty，而平滑 operational cost 几乎没有区别：说明 benchmark 太脆弱；
4. A1 中只需给模型加一个简单 lag state 就能消除所有 discrepancy gap：说明我们应把问题重新定位为 model selection，而不是 calibration-aware planning；
5. Bayes oracle 与 clairvoyant oracle 本身差距巨大：说明环境中的信息限制确实很强；此时不能把 clairvoyant gap 误写成算法缺陷。

## 8. 建议的实现顺序与交付物

第一轮代码实现建议只做 scalar：

1. 复现 static \(T=2\) 的 CE、KH-AD 和 numerical exact/near-exact cost curves；
2. 包装成不可重放的 \(T_{\mathrm{dep}}=300\) deployment loop；
3. 加 slow drift 与 fixed jumps；
4. 加 KH-AD-RW、CE-RW、Bayes oracle 和 clairvoyant；
5. 加 sparse physical data、non-smooth switching cost 和 multimodal regime 的 scalar stress tests；
6. 输出总成本表、paired regret、change-aligned recovery plot、posterior plot、physical/calibration transition 数和 twin-call budget 曲线；
7. scalar 机制验证通过后，再实现 CartPole matched family；最后才加入 discrepancy。

建议首轮代码产物：

```text
benchmarks/
  scalar_dual/
  cartpole_twin/
controllers/
  ce.py
  kh_ad.py
  oracle.py
experiments/
  configs/
  run_scalar.py
  run_cartpole.py
reports/
  figures/
  tables/
```

## 9. 当前不需要阻塞实现、但之后要确认的选择

以下默认值足以开始，不需要现在停下来确认：

- benchmark A 使用 actuator gain 作为一维 \(\theta_t\)，而非一开始同时漂移 mass、length 和 friction；
- primary objective 用 expected realized cumulative cost，quantile/CVaR 先作为 secondary metric；
- scalar 中同时报告 \(\lambda_\Delta=0\) 和 \(0.05\)，再加入一个 non-smooth migration cost stress test；
- sparse physical data 作为单独 stress test，先扫 \(m\in\{1,5,10,20\}\)；
- CartPole 先做 matched family，再加 actuator lag + Coulomb friction；
- oracle 同时保留 Bayes oracle/reference 和 clairvoyant ceiling；
- 第一轮不用 learned surrogate，也不引入 proposed method。

真正需要在 CartPole 正式大规模运行前确认的是：reference schedule 是否对应我们想讲的应用场景，以及 switching cost 应解释为 force ramping、recipe migration 还是 energy/transient risk。这会影响论文叙事，但不妨碍先把 scalar benchmark 跑通。

## 10. 一句话结论

这两个 benchmark 的组合是合理的：**scalar toy 负责给出 near-exact、可归因的 mechanism evidence；CartPole digital twin 负责证明该 gap 在连续控制、迁移成本、不可重放 physical trajectory 和 model discrepancy 下仍然存在。** 最关键的不是先让新方法 beat baselines，而是先证明：在静态设定下原始 dual control 正常工作，而当 calibration state 漂移或跳变时，它具体在哪一段 physical cumulative reward 上付出代价。
