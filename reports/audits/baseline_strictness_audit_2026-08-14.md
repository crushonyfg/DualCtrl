# Baseline strictness audit: KH 2016, Arcari 2020, Bogunovic 2016

Date: 2026-08-14
Repository: `/mnt/bn/feed-quality-training/user/yxu/DualCtrl`
Scope: audit only. No implementation changes were made. This report reviews current baseline implementations for strictness against the cited papers, with special focus on CartPole strictness and oracle wording.

## Executive verdict

| Baseline / environment | Verdict | Reason |
|---|---:|---|
| Klenske-Hennig 2016 scalar Sec. 6.1 reproduction in `controllers/kh_strict.py` | PASS with caveats | Implements the scalar linear-Gaussian equations and records the paper's unstated `x0` ambiguity. Needs stronger numeric comparison to Eq. (9) sampling/reference curve and more equation-level tests for the AD Riccati/covariance term. |
| Klenske-Hennig scalar official deployment in `controllers/official.py` | FAIL as strict KH-AD reproduction | Uses fantasy quadrature plus exploitation recursion, not the explicit KH Sec. 4 augmented covariance/Riccati objective implemented in `kh_strict.py`. It is a KH-inspired planner, not a strict paper baseline. |
| Klenske-Hennig CartPole official runner/controller | FAIL | `experiments/run_official_cartpole.py` uses `KHGPControllerCartPole`, whose file states it is a scaffold/reference and not a strict reproduction. The controller uses random Fourier features, residual GP, fantasy-tail planning, and heuristic variance penalties rather than the full KH GP AD algorithm. |
| Arcari 2020 scalar | PARTIAL PASS | The scalar scenario-tree builder has explicit dual/exploitation split and useful tests for Eqs. (5)-(10), but it uses deterministic sigma/Gauss-Hermite scenarios, finite-grid enumeration, `nm=1`, scalar-specific Gaussian belief, and finite-action rather than the paper's continuous NLP. Must be labeled a finite-action scalar specialization. |
| Arcari 2020 CartPole | FAIL | No explicit CartPole scenario tree object/tests; root stage handling differs from scalar implementation; posterior update is a pseudo-observation grid search; structural modes absent; noise scenarios are handcrafted sparse points; finite-grid control only. Strict paper claim is not supported. |
| Bogunovic 2016 TV-GP-UCB core | PASS for finite-action core equations | Implements Markov time-space covariance, finite-domain beta schedule, exact GP posterior, UCB acquisition, and LCB sign flip; tests cover these. |
| Bogunovic 2016 control-task adapters | PARTIAL / label carefully | The adapter correctly uses realized immediate costs only, but mapping CartPole/scalar receding-control contexts to a finite-action bandit is a benchmark reduction, not in the paper. It is strict only as “Bogunovic finite-action TV-GP-LCB applied to realized one-step costs”. |
| Oracle `oracle_trend` | PASS only as pathwise finite-grid short-horizon reference | It enumerates over the same configured discrete action grid and realized future parameter/noise path for the planning horizon. It must not be called theoretical optimal, Bayes optimal, or exact continuous optimum. Reports/regrets should say “pathwise finite-grid oracle/reference” or “clairvoyant finite-grid MPC reference”. |

## Files audited

- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/controllers/kh_strict.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/controllers/kh_gp.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/controllers/official.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/controllers/cartpole.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/controllers/tv_gp_ucb.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/experiments/reproduce_kh_section6.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/experiments/run_official_scalar.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/experiments/run_official_cartpole.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/experiments/run_official_suite.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/benchmarks/scalar_dual/*.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/benchmarks/cartpole_twin/*.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/tests/test_kh_section61.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/tests/test_arcari_smpc.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/tests/test_tv_gp_ucb.py`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/benchmark_design.md`
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/reports/tables/official_final/official_report.md`

## Detailed checklist

### 1. Klenske-Hennig 2016

#### 1.1 Scalar Sec. 6.1 strict reproduction (`controllers/kh_strict.py`, `experiments/reproduce_kh_section6.py`)

Checklist:

- PASS: Scalar dynamics match the paper setting: `x[k+1] = a x[k] + b u[k] + xi[k]`, known `a`, unknown constant `b`.
- PASS: Constants match text in Sec. 6.1: `a=1`, true `b=2`, prior `N(1,10)`, `Q=1e-1`, `R=0`, `W=1`, `Lambda=1`, terminal weight `1`, `T=2`.
- PASS with caveat: `x0` ambiguity is explicitly documented and written to CSV; default `x0=1.0` is a reproduction choice, not a stated paper constant.
- PASS: Eq. (7)-style scalar Gaussian posterior update is implemented and tested.
- PASS: CE and OF/cautious one-step laws are implemented and tested for a simple numeric case.
- PASS with caveat: AD objective is implemented as nominal trajectory plus quadratic uncertainty cost using augmented perturbation state `z=(x,b)` and a Riccati-like recursion.
- FAIL / missing evidence: Tests do not verify `_dual_uncertainty_cost` against a hand-derived numeric case, a published curve, or a trusted reference implementation.
- FAIL / missing evidence: The test only checks nonzero probing minima; it does not check that KH-AD is closer to Eq. (9) sampling/exact-dual landscape than CE/OF, which is the key Sec. 6.1 reproduction claim in `benchmark_design.md`.
- FAIL / missing evidence: `exact_sampling_dual_cost` is optional in the writer and not exercised by a test with fixed seed/tolerance.
- CAVEAT: Action grid defaults to `[-1,1]`; paper’s original problem may not impose this same action bound. This is acceptable only if reported as a bounded finite-grid reproduction/harness.
- CAVEAT: `_dual_uncertainty_cost` includes a `0.5 * dual` factor. This may be consistent with the paper’s quadratic cost convention, but no test currently pins that factor to the paper equations.

Required fixes before strict claim:

1. Add a deterministic regression test comparing `kh_ad_scalar_cost`, CE, OF, and `exact_sampling_dual_cost` over a small grid with fixed seed and tolerance; assert the qualitative relation required by Sec. 6.1.
2. Add a hand-computable test for `_dual_uncertainty_cost` at `T=1` or `T=2`, including the `0.5` factor.
3. Report `x0` as a reproduction assumption everywhere; never present it as from the paper.
4. If action limits are used, state “bounded finite-grid reproduction” rather than implying the continuous paper optimization exactly.

#### 1.2 Official scalar KH deployment (`KHDualControlScalar` in `controllers/official.py`)

Checklist:

- PASS: Uses a Gaussian belief and branch-specific fantasy posterior updates, so it has a dual-control flavor.
- FAIL: It does not call or reuse `kh_ad_scalar_cost` / KH Sec. 4 augmented Riccati objective from `controllers/kh_strict.py`.
- FAIL: Root action value is computed by Gauss-Hermite fantasies plus `_scalar_exploitation_value`; this is a stochastic lookahead approximation/substitution, not the KH approximate dual cost equation.
- FAIL: `_scalar_exploitation_value` freezes belief mean/variance during tail exploitation and uses expected next-square terms, but does not implement the KH augmented covariance/Riccati dual term.
- FAIL as paper strictness: `predict()` can add Gaussian random-walk process variance through `GaussianBelief.process_var`, while KH Sec. 6.1 assumes a deterministic unknown parameter in the static setting. In current constructors process variance defaults to zero, but the abstraction permits a different model without explicit labeling.

Required fixes:

1. If the official scalar baseline is meant to be KH 2016, route it through the strict KH objective or rename it “KH-inspired fantasy lookahead”.
2. Add a test that official scalar KH and `kh_strict` agree in the Sec. 6.1 `T=2` case over the same action grid, if strict equivalence is intended.
3. Document any continuous-deployment dynamic-parameter extension as a benchmark extension, not as the paper algorithm.

#### 1.3 CartPole KH (`KHGPControllerCartPole`, `KHDualControlCartPole`, runner selection)

Current facts:

- `controllers/kh_gp.py` explicitly states: “Remaining gap … not fully specified … scaffold/reference implementation … should not be reported as a strict reproduction of the full GP AD algorithm.”
- `experiments/run_official_cartpole.py` imports and uses `KHGPControllerCartPole`, not `KHDualControlCartPole` from `controllers/official.py`.
- `KHDualControlCartPole` exists in `controllers/official.py` but is not used by the official CartPole runner.

Checklist for CartPole strictness:

- FAIL: CartPole is not a problem from KH 2016. A CartPole result can only be a KH-style extension unless the full GP dynamics formulation from Sec. 5.2 is implemented and clearly mapped.
- FAIL: `KHGPControllerCartPole` is a scaffold. It uses finite random Fourier features and Bayesian linear updates for residual dynamics, then a fantasy-tail planner. This is not the KH Sec. 4 augmented covariance/Riccati approximate dual objective for GP dynamics.
- FAIL: The planner adds heuristic uncertainty via `value += sum(var_res)` inside `_exploitation_value`; this is not a paper equation.
- FAIL: Fantasy branching uses a single Gauss-Hermite node shared across all output dimensions as `fantasy_next = pred_mean + sqrt(2 var_res) * node`; this is a moment/heuristic branch, not the paper’s posterior equation (16)-(17) propagated through the AD cost.
- FAIL: The feature map is random Fourier features, while KH Sec. 5.2.1 discusses finite basis approximations, but the downstream AD recursion is not implemented.
- FAIL: CartPole belief over actuator gain in `KHDualControlCartPole` is updated via `_cartpole_theta_pseudo_observation`, a grid-search pseudo-measurement over theta; this is a substitution for an EKF/UKF/particle/filter likelihood and is not KH GP dynamics.
- FAIL: `cartpole_step` supports actuator lag/friction in the physical dynamics; official controllers are forced to nominal dynamics in gap settings. This is a benchmark design choice, not a KH paper condition.
- FAIL / missing tests: No CartPole test asserts any KH equation, GP posterior equation, augmented covariance recursion, or equivalence to scalar Sec. 6.1 in a reduced limit.

Required fixes:

1. Do not label CartPole KH output as a strict KH 2016 baseline. Use “KH GP scaffold” or “KH-inspired GP fantasy planner” until fixed.
2. Decide the intended strict target: either (a) KH scalar only, with CartPole excluded from strict literature claims, or (b) implement full KH GP approximate dual control for CartPole residual dynamics.
3. Replace pseudo-observation grid search with a declared likelihood/filter shared across methods, and test it.
4. Remove heuristic variance penalty or derive and test it from the KH AD uncertainty term.
5. Add CartPole strictness tests only after a formal equation-to-code mapping exists.

### 2. Arcari et al. 2020 Dual Stochastic MPC

#### 2.1 Scalar Arcari (`ArcariDualSMPCScalar`, `build_arcari_scalar_scenario_tree`)

Checklist:

- PASS: Explicit scenario tree is built for the dual part.
- PASS: Node probabilities are propagated and tested; levels sum to one for `nm=1`.
- PASS: Child state equation matches scalar sampled parameter/noise transition.
- PASS: Child information state is branch-specific and tested.
- PASS: Objective is decomposed into dual cost and exploitation cost and tested against manual sums.
- PASS with label: Uses finite action grid enumeration instead of continuous nonlinear programming; acceptable only as a finite-action benchmark specialization.
- PASS with label: Uses `nm=1`, no structural modes. Acceptable for a no-mode scalar specialization, but not a full paper reproduction with model-mode discrimination.
- PARTIAL: Parameter scenarios are deterministic Gauss-Hermite/sigma points from `_sigma_scenarios`; Arcari describes sampling-based scenarios. This is a quadrature substitution. It may be valid numerically, but must be disclosed.
- PARTIAL: Noise scenarios are `[-sqrt(var),0,+sqrt(var)]` with weights `[1/6,2/3,1/6]`, a handcrafted cubature rule rather than random samples.
- FAIL / unused argument: `_arcari_scalar_action(..., rng)` accepts `rng` but does not use it. This confirms deterministic quadrature, not sampled scenarios.

Required fixes:

1. Label as “finite-action scalar specialization with deterministic quadrature scenarios,” not generic strict Arcari DMPC.
2. Either remove unused `rng` or implement actual sampled scenarios with seed-controlled reproducibility; if deterministic quadrature remains, document it as a deliberate substitution.
3. Add tests for non-root action optimization in the dual tree and for finite-grid root optimality.
4. Add a test for the degenerate cases `L=0`, `L=N`, zero parameter variance, and zero process variance.

#### 2.2 CartPole Arcari (`ArcariDualSMPCCartPole`)

Checklist:

- FAIL: No explicit `ArcariCartPoleScenarioTree` object analogous to scalar; tests cannot inspect Eq. (5)-(10) structure.
- FAIL: `_arcari_cartpole_action` evaluates `_arcari_cartpole_child_expectation` for each root action but does not add a separate root stage in the root loop; root stage is embedded in child expectation. This may be mathematically okay, but it is not exposed/tested as a unified tree objective.
- FAIL: Posterior update uses `_cartpole_theta_pseudo_observation`, a coarse grid search over theta candidates, then GaussianBelief update with `x=0,u=1`. This is a heuristic pseudo-measurement, not Arcari Bayesian information-state update for nonlinear dynamics.
- FAIL: Structural mode probabilities `p(M|I)` are absent; implementation is `nm=1` only.
- FAIL: Noise scenarios are sparse hand-designed state perturbations from `smpc_noise_std`, not the paper’s sampled disturbance/parameter scenario construction.
- FAIL: Controls are finite-grid actions; paper solves an optimization problem over constrained continuous inputs.
- FAIL / missing tests: No CartPole Arcari test covers tree branching, probabilities, branch-specific posterior updates, exploitation split, failure handling, or root/tail action consistency.

Required fixes:

1. If strict Arcari CartPole is required, build an explicit scenario tree data structure with node probabilities, sampled theta/noise, branch beliefs, and dual/exploitation costs.
2. Replace theta pseudo-observation with a documented likelihood/filter for the nonlinear transition; share it where fairness requires.
3. Add tests mirroring `test_arcari_smpc.py` for CartPole.
4. Label finite-grid enumeration and deterministic cubature as benchmark substitutions.

### 3. Bogunovic, Scarlett, and Cevher 2016 TV-GP-UCB

#### 3.1 Core GP bandit implementation (`controllers/tv_gp_ucb.py`)

Checklist:

- PASS: Implements Markov GP evolution covariance as `(1 - epsilon) ** (|t-s|/2) * k_x(x,x')`.
- PASS: Enforces one-based positive paper time indices.
- PASS: Implements exact GP posterior mean and latent variance with `K + sigma^2 I`.
- PASS: Implements finite-domain beta schedule `2 log(|D| pi_t / delta)` with `pi_t = pi^2 t^2 / 6`.
- PASS: `TVGPUCB` selects reward-maximizing UCB.
- PASS: `TVGPLCB` sign-flips to cost-minimizing LCB.
- PASS: Tests cover kernel values, posterior single-observation closed form, beta formula, UCB/LCB acquisition, and realized-cost adapter update semantics.
- CAVEAT: Bogunovic 2016 also includes regret/batch/reset discussions; only the finite-action TV-GP-UCB core and an auxiliary R-GP-UCB reset implementation are present.

Required fixes:

1. Add tests for time-varying posterior with multiple observations at different times, including temporal asymmetry mistakes.
2. Add tests for `epsilon=0` and `epsilon=1` edge cases.
3. Add a reset/batch behavior test if `RGPUCB` is reported.

#### 3.2 Control adapters (`TVGPLCBScalar`, `TVGPLCBCartPole`)

Checklist:

- PASS: Acquisition uses only realized immediate stage costs, not rollout terminal costs or simulator-predicted costs.
- PASS: Feature mapping includes context and action; finite feasible action set is rebuilt each decision while preserving posterior history.
- PARTIAL: Treating each current context/action pair as a finite-action bandit arm is an application mapping, not a control algorithm in Bogunovic 2016.
- PARTIAL: Updates use immediate stage costs only. If reported against cumulative-control baselines, this is intentionally myopic and should not be described as MPC or dual control.
- FAIL / missing tests: CartPole adapter has no test that the realized cost is computed using the observed failed state from `observe()` and that no terminal/failure heuristic leaks into acquisition before observation.
- FAIL / missing tests: Scalar adapter has no test for pending-cost finalization across rollout ordering beyond the generic adapter test.

Required fixes:

1. Label as “Bogunovic finite-action TV-GP-LCB on realized one-step costs.”
2. Add CartPole-specific tests proving no nominal rollout/terminal heuristic enters acquisition or update.
3. Add test that `observe()`/`act()` ordering records exactly one cost per physical step.

### 4. CartPole-specific strictness audit

CartPole is the highest-risk area.

#### PASS items

- PASS: Dynamics and costs are explicit and reproducible in `benchmarks/cartpole_twin/dynamics.py`, `costs.py`, and `env.py`.
- PASS: Physical environment separates nominal/no-gap and gap settings; observations can be subsampled.
- PASS: Oracle enumerates over the same discrete action grid and finite planning horizon used by baselines.

#### FAIL items / substitutions

- FAIL: KH CartPole is a scaffold, not strict KH GP AD.
- FAIL: Arcari CartPole lacks explicit inspected tree and uses pseudo-observation filter.
- FAIL: CartPole posterior update is not shared in a principled way across KH/Arcari. Both use `_cartpole_theta_pseudo_observation` in `controllers/official.py`, while `KHGPControllerCartPole` uses a residual GP with random Fourier features. This changes the information state and makes literature-method comparisons hard to interpret.
- FAIL: `experiments/run_official_cartpole.py` labels output baseline `kh_dual_control` even though it instantiates `KHGPControllerCartPole`, whose own docstring says not strict.
- FAIL: No CartPole tests for KH, Arcari, oracle pathwise optimality, or absorber/failure treatment in oracle vs environment.
- FAIL: The environment’s absorbing failure is not exactly mirrored in `_oracle_cartpole_value`: when a failure occurs, oracle adds failure-cost multiples and stops recursion, but terminal cost treatment and state absorption are not explicitly tested against environment rollout semantics.
- FAIL: `CartPoleEnvConfig.reference_segment` can differ from `reference_position(t)` calls in controllers/oracle, which use the default segment unless passed. Current default matches, but this is not guarded by tests.
- FAIL: `cartpole_step` has actuator lag state `prev_force`; most planning routines call it without propagating `prev_force` through multi-step planning except the older `_shooting_mpc_action`. Official CartPole oracle and baselines ignore force-lag state in planning even when the physical gap uses lag; that is acceptable for non-oracle baselines in gap settings but means the oracle is not optimal for the full lagged physical state unless lag is included in oracle state/action recursion.

Required CartPole fixes before any strict paper claims:

1. Rename report labels or controller names so CartPole KH is not presented as strict KH 2016.
2. Add an explicit CartPole oracle test: for a tiny horizon and action grid, brute-force all discrete action sequences and verify `_oracle_cartpole_action` picks the minimizing first action under the same dynamics, noise path, reference, failure, and terminal rules.
3. If the oracle is used in lag/friction gap settings, either include lag/friction state in oracle planning or label it as a nominal/no-gap finite-grid oracle only; otherwise it is not pathwise optimal for the true physical system.
4. Implement and test a shared nonlinear Bayesian theta filter or explicitly state each baseline’s different filter assumption.
5. Add Arcari CartPole scenario-tree tests before claiming Arcari strictness.
6. Add KH CartPole equation-mapping tests only after implementing full KH GP AD.

### 5. Oracle definition expectations

Current implementation:

- `OracleTrendScalar` knows the future realized `b_path`, discrepancy term, and noise path for the configured planning horizon and minimizes over `config.action_grid` recursively.
- `OracleTrendCartPole` knows future realized `theta_path` and noise path for the configured planning horizon and minimizes over `config.action_grid` recursively.

Verdict:

- PASS as a pathwise finite-grid, finite-horizon clairvoyant/reference planner over the same discrete action class.
- FAIL if called theoretical optimal, exact continuous optimum, Bayes oracle, or globally optimal over the full deployment horizon.

Required wording:

- Use: “pathwise finite-grid oracle”, “discrete-action clairvoyant MPC reference”, or “same-action-grid short-horizon oracle/reference”.
- Avoid: “theoretical optimal oracle”, “Bayes optimal oracle”, “exact oracle”, “true optimum”, unless a pathwise/discrete dynamic program over the same action class and full relevant state/noise/path semantics is implemented and tested.

Current docs status:

- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/reports/tables/official_final/official_report.md` already marks old results invalid and explicitly says `oracle_trend` is not theoretical optimal. PASS.
- `/mnt/bn/feed-quality-training/user/yxu/DualCtrl/benchmark_design.md` correctly distinguishes Bayes oracle and clairvoyant oracle/reference and warns against exact CartPole oracle claims. PASS.
- `run_official_scalar.py` and `run_official_cartpole.py` compute `oracle_regret` against `oracle_trend`. PARTIAL: column name is convenient but should be renamed or footnoted as `pathwise_finite_grid_oracle_regret` if published.

Oracle tests missing:

1. Scalar brute-force pathwise finite-grid optimality test for tiny horizon/action grid.
2. CartPole brute-force pathwise finite-grid optimality test for tiny horizon/action grid.
3. Tests that a baseline cannot beat the oracle when both are evaluated in exactly the same deterministic pathwise finite-grid setting and horizon/action class; if it can, oracle semantics are mismatched.
4. Tests for failure/absorbing-state semantics in CartPole oracle.
5. Tests that oracle label/report text does not contain “theoretical optimal” or “exact oracle” unless a stricter oracle is implemented.

### 6. Tests missing by baseline

| Baseline | Missing tests required for strictness |
|---|---|
| KH scalar strict | Eq. (9) sampling/reference comparison; hand-derived AD uncertainty term; action-grid/bound caveat; official-vs-strict equivalence if official uses KH name. |
| KH scalar official | Agreement with `kh_strict` in Sec. 6.1 or rename; tests proving no fantasy substitution if strict claim remains. |
| KH CartPole | Full KH GP posterior equations; augmented covariance/Riccati AD recursion for GP dynamics; no heuristic variance penalty; shared filter; reduced scalar-limit test. |
| Arcari scalar | Root finite-grid optimality; non-root action optimization; `L=0`, `L=N`, zero-variance edge cases; deterministic quadrature vs random sampling label. |
| Arcari CartPole | Explicit scenario tree counts/weights/parent links; branch state; posterior update; dual/exploitation objective; root stage consistency; failure semantics. |
| TV-GP-UCB core | Multi-time posterior; `epsilon` edge cases; R-GP-UCB reset if reported. |
| TV-GP-LCB adapters | CartPole realized-cost-only update; act/observe ordering; no rollout/terminal heuristic in acquisition. |
| Oracle | Brute-force pathwise finite-grid optimality for scalar/CartPole; failure/terminal/absorbing semantics; label-regression tests. |

## Required fixes summary

1. CartPole: stop presenting current KH CartPole as strict KH 2016. The code itself says scaffold.
2. CartPole: stop presenting current Arcari CartPole as strict Arcari 2020 until explicit tree and branch belief tests exist.
3. Scalar KH: keep `kh_strict.py` as the strict reference, but add stronger numeric equation tests before relying on it as a paper reproduction gate.
4. Official scalar KH: either use `kh_strict.py` objective or rename the controller as KH-inspired fantasy lookahead.
5. Bogunovic: core is strict finite-action TV-GP-UCB/LCB; adapters must be described as an application mapping to realized one-step control costs.
6. Oracle: use only finite-grid/pathwise/clairvoyant-reference wording; do not use theoretical optimal unless full pathwise/discrete optimality over the same action class and state semantics is implemented and tested.
7. Published result tables should rename or footnote `oracle_regret` as regret to `oracle_trend`, a pathwise finite-grid short-horizon reference, not Bayes/theoretical optimal.

## Final audit conclusion

The repository has made good progress on scalar KH strict reproduction and the Bogunovic finite-action GP bandit core. However, the current CartPole baselines are not strict implementations of KH 2016 or Arcari 2020. They include scaffolds, pseudo-observation filters, finite-grid substitutions, deterministic cubature, fantasy planning, and heuristic uncertainty penalties. These may be useful benchmark baselines if labeled honestly, but they should not be claimed as paper-strict CartPole reproductions.

The oracle implementation is acceptable as a same-grid pathwise reference, but only with limited wording. It must not be called theoretical optimal, exact oracle, or Bayes oracle unless a tested pathwise/discrete optimal solver over the same action class and full physical state semantics is implemented.
