"""Static two-step scalar sanity curves inspired by Klenske-Hennig.

The exact KH approximation from the paper should be checked carefully before
using these curves as a publication-quality reproduction. This module provides
first-pass CE, fantasy-update approximate dual, and grid Bayes references for
implementation sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.filters import GaussianBelief, GridBelief
from controllers.scalar import (
    ApproxDualController,
    CertaintyEquivalentController,
    GridBayesController,
    ScalarPlannerConfig,
    _grid_one_step_action,
)


@dataclass(frozen=True)
class StaticCurve:
    u0: np.ndarray
    ce_cost: np.ndarray
    approx_dual_cost: np.ndarray
    grid_bayes_cost: np.ndarray


def compute_static_two_step_curve(action_grid_size: int = 121) -> StaticCurve:
    cost = ScalarCost(ScalarCostConfig(energy_weight=1.0, switch_weight=0.0, terminal_weight=1.0))
    planner = ScalarPlannerConfig(action_grid_size=action_grid_size, process_var=0.1)
    u0_grid = planner.action_grid
    ce = CertaintyEquivalentController("ce_static", GaussianBelief(mean=1.0, var=10.0), cost, planner)
    ad = ApproxDualController("ad_static", GaussianBelief(mean=1.0, var=10.0), cost, planner)
    b_grid = np.linspace(0.3, 3.7, 41)
    grid_belief = GridBelief.normal_prior(b_grid, mean=1.0, var=10.0)
    bayes = GridBayesController("grid_bayes_ref", grid_belief, cost, planner)
    x0 = 1.0
    prev_u = 0.0
    ce_cost = np.array([_forced_first_action_cost(ce, x0, prev_u, float(u), cost, planner) for u in u0_grid])
    ad_cost = np.array([_forced_first_action_cost(ad, x0, prev_u, float(u), cost, planner) for u in u0_grid])
    bayes_cost = np.array([_forced_grid_bayes_cost(bayes, x0, prev_u, float(u), cost, planner) for u in u0_grid])
    return StaticCurve(u0=u0_grid, ce_cost=ce_cost, approx_dual_cost=ad_cost, grid_bayes_cost=bayes_cost)


def _forced_first_action_cost(controller, x: float, prev_u: float, u0: float, cost: ScalarCost, planner: ScalarPlannerConfig) -> float:
    immediate = cost.stage(x, u0, prev_u).total
    sigma = np.sqrt(planner.process_var + u0 * u0 * controller.belief.var)
    points = [-1.0, 0.0, 1.0]
    weights = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
    total = immediate
    for w, p in zip(weights, points):
        next_x = x + controller.belief.mean * u0 + p * sigma
        fantasy = controller.belief.copy()
        if isinstance(controller, ApproxDualController):
            fantasy.update(x, u0, next_x, planner.process_var)
        temp = CertaintyEquivalentController("tmp", fantasy, cost, planner)
        u1 = temp.act(next_x, u0)
        pred_x2 = next_x + fantasy.mean * u1
        total += w * (cost.stage(next_x, u1, u0).total + cost.terminal(pred_x2).total)
    return float(total)


def _forced_grid_bayes_cost(controller: GridBayesController, x: float, prev_u: float, u0: float, cost: ScalarCost, planner: ScalarPlannerConfig) -> float:
    immediate = cost.stage(x, u0, prev_u).total
    eps_points = np.array([-1.0, 0.0, 1.0]) * np.sqrt(planner.process_var)
    eps_weights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    total = immediate
    for b, bw in zip(controller.belief.grid, controller.belief.weights):
        for ew, eps in zip(eps_weights, eps_points):
            next_x = x + b * u0 + eps
            fantasy = controller.belief.copy()
            fantasy.update(x, u0, next_x, planner.process_var)
            u1 = _grid_one_step_action(next_x, u0, fantasy, cost, planner)
            pred_x2s = next_x + fantasy.grid * u1
            expected_terminal = float(np.sum(fantasy.weights * (planner.process_var + pred_x2s * pred_x2s)))
            total += bw * ew * (cost.stage(next_x, u1, u0).total + expected_terminal)
    return float(total)
