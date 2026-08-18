"""Strict Klenske-Hennig scalar approximate dual control reproduction.

This module implements the scalar linear-Gaussian setting used in Klenske &
Hennig (2016), Sec. 3.1 and Sec. 6.1:

    x[k+1] = a x[k] + b u[k] + xi[k],       xi[k] ~ N(0, Q)
    b ~ N(mu, sigma^2),                     R = 0 in Sec. 6.1

The approximate-dual (AD) objective follows Sec. 4: for each candidate root
control u[k], compute the certainty-equivalent nominal trajectory under the
mean parameter, then add the quadratic perturbation/uncertainty cost from the
augmented covariance filtering recursion

    z = (x, b),   A_tilde = [[a, u_j], [0, 1]],   B_tilde = [[mu], [0]],

and the augmented Riccati recursion in the paragraph containing Eq. (13) and
Eq. (J^d) in the paper text.  CE and OF/cautious functions are included only as
paper reference comparisons from Sec. 3.1, not as new baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.filters import GaussianBelief


@dataclass(frozen=True)
class KHSection61Constants:
    """Constants stated in Klenske-Hennig Sec. 6.1.

    Sec. 6.1 specifies a=1, true b=2, p(b)=N(1,10), Q=1e-1, R=0,
    W=1, Lambda=1, T=2.  The initial state x0 is not stated in the paper text;
    reproduction scripts default to x0=1 because the plotted dual cost has a
    non-zero probing minimizer, while x0=0 makes probing valueless for T=2.
    """

    a: float = 1.0
    true_b: float = 2.0
    prior_mean: float = 1.0
    prior_var: float = 10.0
    process_var: float = 1e-1
    obs_var: float = 0.0
    state_weight: float = 1.0
    energy_weight: float = 1.0
    terminal_weight: float = 1.0
    horizon: int = 2
    x0_default: float = 1.0


@dataclass(frozen=True)
class KHScalarADConfig:
    a: float = 1.0
    process_var: float = 0.1
    obs_var: float = 0.0
    horizon: int = 2
    action_low: float = -1.0
    action_high: float = 1.0
    action_grid_size: int = 401

    @property
    def action_grid(self) -> np.ndarray:
        return np.linspace(self.action_low, self.action_high, self.action_grid_size)


class KHScalarADController:
    name = "kh_dual_control"

    def __init__(self, belief: GaussianBelief, cost: ScalarCost, config: KHScalarADConfig):
        self.belief = belief
        self.cost = cost
        self.config = config

    @property
    def belief_mean(self) -> float:
        return self.belief.mean

    @property
    def belief_var(self) -> float:
        return self.belief.var

    def predict(self) -> None:
        # KH Sec. 4 assumes deterministic but unknown parameters.
        pass

    def observe(self, x: float, u: float, observed_next_x: float, obs_var: float) -> None:
        """Gaussian posterior update for b, Eq. (7), with known a.

        The sufficient observation is z = x[k+1] - a x[k] = b u[k] + xi[k].
        """

        posterior_mean, posterior_var = scalar_posterior_update(
            mu=self.belief.mean,
            var=self.belief.var,
            u=u,
            residual=observed_next_x - self.config.a * x,
            noise_var=self.config.process_var + obs_var,
            min_var=self.belief.min_var,
        )
        self.belief.mean = posterior_mean
        self.belief.var = posterior_var

    def act(self, x: float, prev_u: float = 0.0) -> float:
        best_u = 0.0
        best_cost = float("inf")
        for u in self.config.action_grid:
            c = kh_ad_scalar_cost(float(x), float(u), self.belief.mean, self.belief.var, self.cost, self.config, prev_u)
            if c < best_cost:
                best_cost = c
                best_u = float(u)
        return best_u


def make_section61_problem(action_grid_size: int = 401, action_low: float = -1.0, action_high: float = 1.0) -> tuple[KHSection61Constants, ScalarCost, KHScalarADConfig, GaussianBelief]:
    """Build the exact Sec. 6.1 scalar problem constants available from text."""

    c = KHSection61Constants()
    cost = ScalarCost(
        ScalarCostConfig(
            state_weight=c.state_weight,
            energy_weight=c.energy_weight,
            switch_weight=0.0,
            terminal_weight=c.terminal_weight,
        )
    )
    config = KHScalarADConfig(
        a=c.a,
        process_var=c.process_var,
        obs_var=c.obs_var,
        horizon=c.horizon,
        action_low=action_low,
        action_high=action_high,
        action_grid_size=action_grid_size,
    )
    belief = GaussianBelief(mean=c.prior_mean, var=c.prior_var)
    return c, cost, config, belief


def scalar_posterior_update(mu: float, var: float, u: float, residual: float, noise_var: float, min_var: float = 1e-12) -> tuple[float, float]:
    """Closed-form Gaussian update from Eq. (7).

    residual is b*u + xi, i.e. x[k+1] - a*x[k].  The update variance is
    sigma^2 Q / (u^2 sigma^2 + Q); the mean is the corresponding Kalman update.
    """

    if abs(u) < 1e-12:
        return float(mu), max(float(var), min_var)
    pred_var = max(float(var), min_var)
    q = max(float(noise_var), min_var)
    innovation_var = u * u * pred_var + q
    gain = pred_var * u / innovation_var
    posterior_mean = float(mu + gain * (residual - u * mu))
    posterior_var = max(float((1.0 - gain * u) * pred_var), min_var)
    return posterior_mean, posterior_var


def ce_control_law(x: float, mu: float, cost: ScalarCost, config: KHScalarADConfig) -> float:
    """Sec. 3.1 CE one-step law: -a*mu*x / (Lambda + mu^2)."""

    denom = cost.config.energy_weight + mu * mu
    return 0.0 if denom <= 0.0 else float(-(config.a * mu * x) / denom)


def of_control_law(x: float, mu: float, var: float, cost: ScalarCost, config: KHScalarADConfig) -> float:
    """Sec. 3.1 OF/cautious one-step law, Eq. (6)."""

    denom = cost.config.energy_weight + var + mu * mu
    return 0.0 if denom <= 0.0 else float(-(config.a * mu * x) / denom)


def kh_ad_scalar_cost(
    x0: float,
    u0: float,
    mu: float,
    var: float,
    cost: ScalarCost,
    config: KHScalarADConfig,
    prev_u: float = 0.0,
) -> float:
    """Approximate-dual cost J_bar + J^d for a forced root action u0."""

    nominal_x, nominal_u = _nominal_trajectory_with_fixed_root(x0, u0, mu, cost, config, prev_u)
    j_bar = _nominal_cost(nominal_x, nominal_u, cost, prev_u)
    j_dual = _dual_uncertainty_cost(nominal_u, mu, var, cost, config)
    return float(j_bar + j_dual)


def ce_scalar_cost(
    x0: float,
    u0: float,
    mu: float,
    cost: ScalarCost,
    config: KHScalarADConfig,
    prev_u: float = 0.0,
) -> float:
    """CE reference cost landscape for a forced root action."""

    nominal_x, nominal_u = _nominal_trajectory_with_fixed_root(x0, u0, mu, cost, config, prev_u)
    return float(_nominal_cost(nominal_x, nominal_u, cost, prev_u))


def of_scalar_cost(
    x0: float,
    u0: float,
    mu: float,
    var: float,
    cost: ScalarCost,
    config: KHScalarADConfig,
    prev_u: float = 0.0,
) -> float:
    """OF/cautious reference landscape without valuing future information.

    This evaluates the expected quadratic cost of the forced root action and a
    myopic OF final action.  The posterior covariance is deliberately *not*
    reduced as a function of u0 here; otherwise this reference would include the
    information-seeking dual effect.  It is only a Sec. 3.1 comparison curve.
    """

    if config.horizon != 2:
        raise ValueError("of_scalar_cost is implemented only for the Sec. 6.1 T=2 reference")
    w = cost.config.state_weight
    lam = cost.config.energy_weight
    wt = cost.config.terminal_weight
    a = config.a
    q = config.process_var
    mean_x1 = a * x0 + mu * u0
    var_x1 = u0 * u0 * var + q
    # Final OF action uses the predicted mean state and the current parameter uncertainty.
    u1 = of_control_law(mean_x1, mu, var, cost, config)
    mean_x2 = a * mean_x1 + mu * u1
    var_x2 = a * a * var_x1 + u1 * u1 * var + q
    return float(w * x0 * x0 + lam * u0 * u0 + w * (mean_x1 * mean_x1 + var_x1) + lam * u1 * u1 + wt * (mean_x2 * mean_x2 + var_x2))


def exact_sampling_dual_cost(
    x0: float,
    u0: float,
    mu: float,
    var: float,
    true_unused_b: float,
    cost: ScalarCost,
    config: KHScalarADConfig,
    num_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """Monte-Carlo approximation of Eq. (9) for the scalar T=2 system.

    The true b=2 stated in Sec. 6.1 defines the simulated physical system in
    rollouts, but Fig. 1's approximately exact dual cost samples b from the
    current belief as in Eq. (9).  The true_unused_b argument is kept explicit so
    callers record the Sec. 6.1 constant without accidentally conditioning the
    Bayesian expectation on it.
    """

    del true_unused_b
    if config.horizon != 2:
        raise ValueError("exact_sampling_dual_cost is implemented only for T=2")
    rng = np.random.default_rng(seed)
    b = rng.normal(mu, np.sqrt(var), int(num_samples))
    xi0 = rng.normal(0.0, np.sqrt(config.process_var), int(num_samples))
    x1 = config.a * x0 + b * u0 + xi0
    residual = b * u0 + xi0
    if abs(u0) < 1e-12:
        mu1 = np.full_like(x1, mu)
        var1 = float(var)
    else:
        denom = u0 * u0 * var + config.process_var
        mu1 = (var * u0 * residual + mu * config.process_var) / denom
        var1 = float(var * config.process_var / denom)
    lam = cost.config.energy_weight
    w = cost.config.state_weight
    wt = cost.config.terminal_weight
    u1 = -(config.a * mu1 * x1) / (lam + var1 + mu1 * mu1)
    terminal_expectation = (config.a * x1 + mu1 * u1) ** 2 + var1 * u1 * u1 + config.process_var
    return float(w * x0 * x0 + lam * u0 * u0 + np.mean(w * x1 * x1 + lam * u1 * u1 + wt * terminal_expectation))


def _nominal_trajectory_with_fixed_root(
    x0: float,
    u0: float,
    b: float,
    cost: ScalarCost,
    config: KHScalarADConfig,
    prev_u: float,
) -> tuple[np.ndarray, np.ndarray]:
    del prev_u  # no switching term in the paper setting; kept for controller API compatibility
    T = config.horizon
    xs = np.zeros(T + 1, dtype=float)
    us = np.zeros(T, dtype=float)
    xs[0] = x0
    us[0] = u0
    if T == 1:
        xs[1] = config.a * xs[0] + b * us[0]
        return xs, us

    # Finite-horizon CE LQR for remaining controls with known b=mu, Sec. 4.1.
    P = cost.config.terminal_weight
    gains = np.zeros(T, dtype=float)
    for j in range(T - 1, 0, -1):
        denom = cost.config.energy_weight + b * b * P
        gains[j] = -(b * P * config.a) / denom if denom > 0 else 0.0
        P = cost.config.state_weight + config.a * config.a * P - (config.a * P * b) ** 2 / denom

    xs[1] = config.a * xs[0] + b * us[0]
    for j in range(1, T):
        us[j] = np.clip(gains[j] * xs[j], config.action_low, config.action_high)
        xs[j + 1] = config.a * xs[j] + b * us[j]
    return xs, us


def _nominal_cost(xs: np.ndarray, us: np.ndarray, cost: ScalarCost, prev_u: float) -> float:
    total = 0.0
    last_u = prev_u
    for j, u in enumerate(us):
        total += cost.stage(float(xs[j]), float(u), float(last_u)).total
        last_u = float(u)
    total += cost.terminal(float(xs[-1])).total
    return float(total)


def _dual_uncertainty_cost(us: np.ndarray, mu: float, var: float, cost: ScalarCost, config: KHScalarADConfig) -> float:
    T = config.horizon
    # Riccati recursion on augmented perturbation state z=(x,b), Sec. 4 Eq. after J^d.
    wtilde = np.diag([cost.config.state_weight, 0.0])
    terminal = np.diag([cost.config.terminal_weight, 0.0])
    ktilde = [np.zeros((2, 2), dtype=float) for _ in range(T + 1)]
    ktilde[T] = terminal
    for j in range(T - 1, -1, -1):
        A = np.array([[config.a, us[j]], [0.0, 1.0]], dtype=float)
        B = np.array([[mu], [0.0]], dtype=float)
        S = float((B.T @ ktilde[j + 1] @ B).item() + cost.config.energy_weight)
        if S <= 1e-12:
            feedback_term = 0.0
        else:
            feedback_term = (ktilde[j + 1] @ B @ B.T @ ktilde[j + 1]) / S
        ktilde[j] = A.T @ (ktilde[j + 1] - feedback_term) @ A + wtilde
        ktilde[j] = 0.5 * (ktilde[j] + ktilde[j].T)

    # Preposterior covariance filtering. Future expected observations do not change means,
    # but they reduce covariance as a deterministic function of candidate controls.
    sig_filt = np.array([[0.0, 0.0], [0.0, var]], dtype=float)
    dual = 0.0
    for j in range(T):
        dual += cost.config.state_weight * sig_filt[0, 0]
        A = np.array([[config.a, us[j]], [0.0, 1.0]], dtype=float)
        q = np.diag([config.process_var, 0.0])
        sig_pred = A @ sig_filt @ A.T + q
        sig_upd = _measurement_update_cov(sig_pred, config.obs_var)
        dual += float(np.trace((sig_pred - sig_upd) @ ktilde[j + 1]))
        sig_filt = sig_upd
    dual += cost.config.terminal_weight * sig_filt[0, 0]
    return 0.5 * float(dual)


def _measurement_update_cov(sig_pred: np.ndarray, obs_var: float) -> np.ndarray:
    H = np.array([[1.0, 0.0]])
    S = float((H @ sig_pred @ H.T).item() + obs_var)
    if S <= 1e-12:
        if sig_pred[0, 0] <= 1e-12:
            out = sig_pred.copy()
            out[0, :] = 0.0
            out[:, 0] = 0.0
            return out
        K = sig_pred @ H.T / sig_pred[0, 0]
    else:
        K = sig_pred @ H.T / S
    out = sig_pred - K @ H @ sig_pred
    out = 0.5 * (out + out.T)
    out[0, 0] = max(out[0, 0], 0.0)
    out[1, 1] = max(out[1, 1], 0.0)
    return out
