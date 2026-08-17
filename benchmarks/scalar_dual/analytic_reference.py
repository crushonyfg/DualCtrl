"""Analytic two-step Gaussian references for the scalar benchmark.

This module is for the B0 sanity check. It compares three root-action cost
landscapes for

    x_{t+1} = x_t + b u_t + w_t,    b ~ N(mu, var), w_t ~ N(0, q).

The curves separate:
- certainty-equivalent planning: ignores parameter uncertainty and learning;
- cautious planning: accounts for parameter variance but not future learning;
- Bayes dual planning: updates the posterior after the fantasy first
  transition before choosing the second action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss

from .filters import GaussianBelief


@dataclass(frozen=True)
class AnalyticScalarConfig:
    x0: float = 1.0
    prior_mean: float = 1.0
    prior_var: float = 10.0
    process_var: float = 0.1
    state_weight: float = 1.0
    action_weight: float = 1.0
    terminal_weight: float = 1.0
    action_low: float = -3.0
    action_high: float = 3.0
    action_grid_size: int = 401
    quadrature_points: int = 21

    @property
    def action_grid(self) -> np.ndarray:
        return np.linspace(self.action_low, self.action_high, self.action_grid_size)


@dataclass(frozen=True)
class AnalyticCurve:
    u0: np.ndarray
    ce_cost: np.ndarray
    cautious_cost: np.ndarray
    bayes_dual_cost: np.ndarray


def compute_analytic_curve(config: AnalyticScalarConfig = AnalyticScalarConfig()) -> AnalyticCurve:
    u0_grid = config.action_grid
    ce_cost = np.array([two_step_cost_no_learning(config.x0, float(u), config.prior_mean, 0.0, config) for u in u0_grid])
    cautious_cost = np.array([two_step_cost_no_learning(config.x0, float(u), config.prior_mean, config.prior_var, config) for u in u0_grid])
    bayes_cost = np.array([two_step_cost_with_learning(config.x0, float(u), GaussianBelief(config.prior_mean, config.prior_var), config) for u in u0_grid])
    return AnalyticCurve(u0=u0_grid, ce_cost=ce_cost, cautious_cost=cautious_cost, bayes_dual_cost=bayes_cost)


def two_step_cost_no_learning(x0: float, u0: float, mean: float, var: float, config: AnalyticScalarConfig) -> float:
    first = config.state_weight * x0 * x0 + config.action_weight * u0 * u0
    pred_mean = x0 + mean * u0
    pred_var = config.process_var + var * u0 * u0
    return first + _expect_over_normal(lambda x1: _one_step_value(x1, mean, var, config), pred_mean, pred_var, config.quadrature_points)


def two_step_cost_with_learning(x0: float, u0: float, belief: GaussianBelief, config: AnalyticScalarConfig) -> float:
    first = config.state_weight * x0 * x0 + config.action_weight * u0 * u0
    pred_mean = x0 + belief.mean * u0
    pred_var = config.process_var + belief.var * u0 * u0

    def branch_value(x1: float) -> float:
        fantasy = belief.copy()
        fantasy.update(x0, u0, x1, config.process_var)
        return _one_step_value(x1, fantasy.mean, fantasy.var, config)

    return first + _expect_over_normal(branch_value, pred_mean, pred_var, config.quadrature_points)


def _one_step_value(x: float, mean: float, var: float, config: AnalyticScalarConfig) -> float:
    # Minimize: state_weight*x^2 + action_weight*u^2
    #           + terminal_weight*E[(x + b u + w)^2].
    denom = config.action_weight + config.terminal_weight * (mean * mean + var)
    if denom <= 0.0:
        u_star = 0.0
    else:
        u_star = -config.terminal_weight * x * mean / denom
    u_star = float(np.clip(u_star, config.action_low, config.action_high))
    expected_next_sq = (x + mean * u_star) ** 2 + var * u_star * u_star + config.process_var
    return (
        config.state_weight * x * x
        + config.action_weight * u_star * u_star
        + config.terminal_weight * expected_next_sq
    )


def _expect_over_normal(fn, mean: float, var: float, n: int) -> float:
    if var <= 1e-14:
        return float(fn(mean))
    nodes, weights = hermgauss(n)
    xs = mean + np.sqrt(2.0 * var) * nodes
    vals = np.array([fn(float(x)) for x in xs])
    return float(np.dot(weights, vals) / np.sqrt(np.pi))
