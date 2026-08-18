import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams, cartpole_step, reference_position
from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers.official import (
    OfficialCartPoleConfig,
    OfficialScalarConfig,
    _cartpole_exploitation_value,
    _cartpole_noise_scenarios,
    _cartpole_theta_pseudo_observation,
    _normal_sigma,
    _scalar_exploitation_value,
    _sigma_scenarios,
    build_arcari_cartpole_scenario_tree,
    build_arcari_scalar_scenario_tree,
)


def _small_problem():
    config = OfficialScalarConfig(
        horizon=3,
        action_low=-1.0,
        action_high=1.0,
        action_grid_size=3,
        process_var=0.04,
        smpc_dual_horizon=2,
        smpc_scenarios=3,
    )
    cost = ScalarCost(ScalarCostConfig(state_weight=1.0, energy_weight=0.1, terminal_weight=1.0))
    belief = GaussianBelief(mean=1.0, var=0.25, process_var=0.0)
    return config, cost, belief


def test_arcari_tree_counts_parent_links_and_weights_eqs_5_7():
    config, cost, belief = _small_problem()
    tree = build_arcari_scalar_scenario_tree(0.4, 0.0, 1.0, belief, cost, config)

    n_gamma = len(_sigma_scenarios(belief.mean, belief.var, config.smpc_scenarios))
    n_w = len(_normal_sigma(config.process_var)[0])
    branch_factor = n_gamma * n_w

    assert tree.branch_factor == branch_factor
    assert tree.node_count_by_depth() == [1, branch_factor, branch_factor**2]
    assert len(tree.nodes) == 1 + branch_factor + branch_factor**2
    assert math.isclose(tree.nodes[0].probability, 1.0)

    for depth in range(tree.dual_horizon):
        for node_id in tree.levels[depth]:
            node = tree.nodes[node_id]
            assert len(node.children) == branch_factor
            child_weight_sum = sum(tree.nodes[child_id].probability for child_id in node.children)
            assert math.isclose(child_weight_sum, node.probability, rel_tol=1e-12, abs_tol=1e-12)
            for child_id in node.children:
                assert tree.nodes[child_id].parent_id == node_id
                assert tree.nodes[child_id].depth == depth + 1

    # Eq. (7), nm=1: pbar_child = p(M|I_child) * pbar_parent * sample_weight.
    # The mode probability is one here, so each level sums to one.
    for level in tree.levels:
        assert math.isclose(sum(tree.nodes[i].probability for i in level), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_arcari_branch_state_and_posterior_updates_eq_5():
    config, cost, belief = _small_problem()
    root_action = 1.0
    x0 = 0.4
    tree = build_arcari_scalar_scenario_tree(x0, 0.0, root_action, belief, cost, config)

    root = tree.nodes[0]
    child = tree.nodes[root.children[0]]

    # Eq. (5), scalar nm=1: x_child = x_parent + gamma_sample * u_parent + w_sample.
    expected_state = root.state + child.parameter_sample * root.action + child.noise_sample
    assert math.isclose(child.state, expected_state, rel_tol=1e-12, abs_tol=1e-12)

    # Branch-specific information state update: the child posterior must equal
    # an independent Bayesian update using that branch's sampled observation.
    expected_belief = belief.copy()
    expected_belief.update(x0, root_action, child.state, config.process_var)
    assert math.isclose(child.belief.mean, expected_belief.mean, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(child.belief.var, expected_belief.var, rel_tol=1e-12, abs_tol=1e-12)

    # Different parameter/noise sampled observations generally induce different
    # branch posteriors; this guards against a fixed-information one-step branch.
    posterior_pairs = {(round(tree.nodes[i].belief.mean, 12), round(tree.nodes[i].belief.var, 12)) for i in root.children}
    assert len(posterior_pairs) > 1


def test_arcari_objective_decomposition_matches_eqs_6_8_10():
    config, cost, belief = _small_problem()
    tree = build_arcari_scalar_scenario_tree(0.4, 0.0, 1.0, belief, cost, config)

    # Eq. (6)/(10) dual part: weighted costs for all dual tree node actions.
    manual_dual = 0.0
    for depth in range(tree.dual_horizon):
        for node_id in tree.levels[depth]:
            node = tree.nodes[node_id]
            manual_dual += node.probability * cost.stage(node.state, node.action, node.prev_action).total
    assert math.isclose(tree.dual_cost, manual_dual, rel_tol=1e-12, abs_tol=1e-12)

    # Eq. (8)/(9) exploitation part: at each depth-L leaf, fix the information
    # state collected during the dual part and solve the branch tail.
    manual_tail = 0.0
    for leaf_id in tree.levels[tree.dual_horizon]:
        leaf = tree.nodes[leaf_id]
        tail = _scalar_exploitation_value(
            leaf.state,
            leaf.prev_action,
            leaf.belief.mean,
            leaf.belief.var,
            cost,
            config,
            config.horizon - tree.dual_horizon,
        )
        manual_tail += leaf.probability * tail
    assert math.isclose(tree.exploitation_cost, manual_tail, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(tree.total_cost, tree.dual_cost + tree.exploitation_cost, rel_tol=1e-12, abs_tol=1e-12)



def _small_cartpole_problem():
    config = OfficialCartPoleConfig(
        horizon=3,
        action_grid_size=3,
        theta_obs_var=0.02,
        smpc_dual_horizon=2,
        smpc_scenarios=3,
        smpc_noise_std=(0.0, 0.01, 0.0, 0.0),
    )
    cost = CartPoleCost(CartPoleCostConfig(failure_cost=100.0))
    dynamics = CartPoleParams()
    belief = GaussianBelief(mean=1.0, var=0.04, process_var=0.0)
    state = np.array([0.02, 0.01, 0.03, -0.02], dtype=float)
    return config, cost, dynamics, belief, state


def test_arcari_cartpole_tree_counts_parent_links_and_weights_eqs_5_7():
    config, cost, dynamics, belief, state = _small_cartpole_problem()
    tree = build_arcari_cartpole_scenario_tree(state, 0.0, 0, 0.5, belief, dynamics, cost, config)

    n_gamma = len(_sigma_scenarios(belief.mean, belief.var, config.smpc_scenarios))
    n_w = len(_cartpole_noise_scenarios(config))
    branch_factor = n_gamma * n_w

    assert tree.dual_horizon == 2
    assert tree.prediction_horizon == 3
    assert tree.branch_factor == branch_factor
    assert tree.node_count_by_depth() == [1, branch_factor, branch_factor**2]
    assert len(tree.nodes) == 1 + branch_factor + branch_factor**2
    assert math.isclose(tree.nodes[0].probability, 1.0)

    for depth in range(tree.dual_horizon):
        for node_id in tree.levels[depth]:
            node = tree.nodes[node_id]
            assert len(node.children) == branch_factor
            child_weight_sum = sum(tree.nodes[child_id].probability for child_id in node.children)
            assert math.isclose(child_weight_sum, node.probability, rel_tol=1e-12, abs_tol=1e-12)
            for child_id in node.children:
                child = tree.nodes[child_id]
                assert child.parent_id == node_id
                assert child.depth == depth + 1
                assert child.mode_probability == 1.0
                assert math.isclose(child.probability, node.probability * child.branch_weight, rel_tol=1e-12, abs_tol=1e-12)

    # Eq. (7), single structural mode: every depth is a probability distribution.
    for level in tree.levels:
        assert math.isclose(sum(tree.nodes[i].probability for i in level), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_arcari_cartpole_sampled_transition_eq_5_and_branch_posterior_updates():
    config, cost, dynamics, belief, state = _small_cartpole_problem()
    root_action = 0.5
    tree = build_arcari_cartpole_scenario_tree(state, 0.0, 0, root_action, belief, dynamics, cost, config)

    root = tree.nodes[0]
    child = tree.nodes[root.children[0]]

    # Eq. (5): x_child = f(x_parent, u_parent, gamma_sample) + process_noise_sample.
    nominal_next, _ = cartpole_step(root.state, root.action, child.parameter_sample, dynamics)
    np.testing.assert_allclose(child.state, nominal_next + child.process_noise_sample, rtol=1e-12, atol=1e-12)

    # Branch-specific information state update from that sampled observation.
    expected_belief = belief.copy()
    theta_obs = _cartpole_theta_pseudo_observation(root.state, root_action, child.state, dynamics)
    expected_belief.update(0.0, 1.0, theta_obs, config.theta_obs_var)
    assert math.isclose(child.belief.mean, expected_belief.mean, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(child.belief.var, expected_belief.var, rel_tol=1e-12, abs_tol=1e-12)

    posterior_pairs = {(round(tree.nodes[i].belief.mean, 12), round(tree.nodes[i].belief.var, 12)) for i in root.children}
    assert len(posterior_pairs) > 1


def test_arcari_cartpole_objective_decomposition_matches_eq_10():
    config, cost, dynamics, belief, state = _small_cartpole_problem()
    tree = build_arcari_cartpole_scenario_tree(state, 0.0, 0, 0.5, belief, dynamics, cost, config)

    # Dual term: expected sampled transition/stage costs for actions on depths 0..L-1.
    manual_dual = 0.0
    for depth in range(1, tree.dual_horizon + 1):
        for node_id in tree.levels[depth]:
            node = tree.nodes[node_id]
            parent = tree.nodes[node.parent_id]
            failed = cost.failed(node.state)
            expected_stage = cost.stage(parent.state, parent.action, parent.prev_action, reference_position(parent.time), failed).total
            assert math.isclose(node.transition_stage_cost, expected_stage, rel_tol=1e-12, abs_tol=1e-12)
            manual_dual += node.probability * expected_stage
    assert math.isclose(tree.dual_cost, manual_dual, rel_tol=1e-12, abs_tol=1e-12)

    # Exploitation term: fixed information at each depth-L leaf for horizon N-L.
    manual_tail = 0.0
    for leaf_id in tree.levels[tree.dual_horizon]:
        leaf = tree.nodes[leaf_id]
        remaining = config.horizon - tree.dual_horizon
        if leaf.failed:
            tail = cost.config.failure_cost * max(remaining, 0)
        else:
            tail = _cartpole_exploitation_value(
                leaf.state,
                leaf.prev_action,
                leaf.time,
                leaf.belief.mean,
                dynamics,
                cost,
                config,
                remaining,
            )
        manual_tail += leaf.probability * tail
    assert math.isclose(tree.exploitation_cost, manual_tail, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(tree.total_cost, tree.dual_cost + tree.exploitation_cost, rel_tol=1e-12, abs_tol=1e-12)
