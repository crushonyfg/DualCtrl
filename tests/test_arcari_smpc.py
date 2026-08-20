import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.cartpole_twin.costs import CartPoleCost, CartPoleCostConfig
from benchmarks.cartpole_twin.dynamics import CartPoleParams, cartpole_step, reference_position
from benchmarks.cartpole_twin.env import CartPoleEnvConfig, CartPolePhysicalEnv
from benchmarks.cartpole_twin.regimes import CartPoleRegimeConfig, generate_theta_path
from benchmarks.scalar_dual.costs import ScalarCost, ScalarCostConfig
from benchmarks.scalar_dual.regimes import ScalarRegimeConfig, generate_b_path
from benchmarks.scalar_dual.filters import GaussianBelief
from controllers.official import (
    ArcariDualSMPCCartPole,
    ArcariDualSMPCScalar,
    ArcariPassiveExploitationCartPole,
    ArcariPassiveExploitationScalar,
    NominalMPCCartPole,
    NominalMPCScalar,
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


def test_fixed_jump_regimes_change_within_short_horizon():
    rng = np.random.default_rng(0)
    scalar = generate_b_path(ScalarRegimeConfig(kind="fixed_jumps", horizon=10), rng)
    cartpole = generate_theta_path(CartPoleRegimeConfig(kind="fixed_jumps", horizon=10), rng)
    assert len(set(np.round(scalar, 12))) == 3
    assert len(set(np.round(cartpole, 12))) == 3


def test_cartpole_absorbing_failure_event_count_is_one_shot():
    cost = CartPoleCost(CartPoleCostConfig(x_failure=0.01, failure_cost=100.0))
    env = CartPolePhysicalEnv(
        CartPoleEnvConfig(initial_state=(0.0, 0.0, 0.0, 0.0), absorbing_failure=True),
        CartPoleParams(),
        cost,
        theta_path=np.ones(3),
        process_noise=np.array([[0.02, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    steps = [env.step(0.0), env.step(0.0), env.step(0.0)]
    assert [step.failed for step in steps] == [True, True, True]
    assert [step.failure_event for step in steps] == [True, False, False]


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


def test_arcari_scalar_continuous_root_action_can_be_non_grid_and_bounded():
    config = OfficialScalarConfig(
        horizon=1,
        action_low=-1.0,
        action_high=1.0,
        action_grid_size=3,
        continuous_actions=True,
        optimizer_xatol=1e-6,
        process_var=0.0,
        smpc_dual_horizon=1,
        smpc_scenarios=1,
    )
    cost = ScalarCost(ScalarCostConfig(state_weight=0.0, energy_weight=0.0, switch_weight=1.0, terminal_weight=0.0))
    belief = GaussianBelief(mean=1.0, var=0.0, process_var=0.0)
    controller = ArcariDualSMPCScalar(belief.copy(), cost, config, seed=0)

    action = controller.act(0.0, 0.37)
    tree = controller.build_tree_for_action(0.0, 0.37, action)

    assert config.continuous_actions
    assert -1.0 <= action <= 1.0
    assert action not in {float(u) for u in config.action_grid}
    assert abs(action - 0.37) < 1e-3
    assert abs(tree.nodes[0].action - action) < 1e-12
    assert len(tree.nodes) == 1 + tree.branch_factor


def test_arcari_cartpole_continuous_root_action_can_be_non_grid_and_bounded():
    config, _, dynamics, belief, state = _small_cartpole_problem()
    cost = CartPoleCost(
        CartPoleCostConfig(
            w_p=0.0,
            w_phi=0.0,
            w_v=0.0,
            w_omega=0.0,
            energy_weight=1.0,
            switch_weight=0.0,
            terminal_p_weight=0.0,
            terminal_phi_weight=0.0,
        )
    )
    config = OfficialCartPoleConfig(
        horizon=1,
        action_grid_size=3,
        continuous_actions=True,
        optimizer_xatol=1e-6,
        theta_obs_var=config.theta_obs_var,
        smpc_dual_horizon=1,
        smpc_scenarios=1,
        smpc_noise_std=(0.0, 0.0, 0.0, 0.0),
    )
    controller = ArcariDualSMPCCartPole(belief.copy(), dynamics, cost, config, seed=0)

    action = controller.act(state, 0.37)
    tree = controller.build_tree_for_action(state, 0.37, action)

    assert config.continuous_actions
    assert -1.0 <= action <= 1.0
    assert action not in {float(u) for u in config.action_grid}
    assert tree.nodes[0].action == action
    assert len(tree.nodes) == 1 + tree.branch_factor


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


def test_arcari_l_zero_degenerates_to_fixed_information_exploitation():
    config, cost, belief = _small_problem()
    config_l0 = OfficialScalarConfig(
        horizon=config.horizon,
        action_low=config.action_low,
        action_high=config.action_high,
        action_grid_size=config.action_grid_size,
        process_var=config.process_var,
        smpc_dual_horizon=0,
        smpc_scenarios=config.smpc_scenarios,
    )
    tree = build_arcari_scalar_scenario_tree(0.4, 0.0, 1.0, belief, cost, config_l0)
    assert tree.dual_horizon == 0
    assert tree.node_count_by_depth() == [1]
    assert tree.dual_cost == 0.0
    assert math.isclose(
        tree.exploitation_cost,
        _scalar_exploitation_value(0.4, 0.0, belief.mean, belief.var, cost, config_l0, config_l0.horizon),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_arcari_l_zero_diagnostic_controller_uses_passive_exploitation():
    config, cost, belief = _small_problem()
    controller = ArcariPassiveExploitationScalar(belief.copy(), cost, config)
    action = controller.act(0.4, 0.0)

    expected = min(
        config.action_grid,
        key=lambda u: build_arcari_scalar_scenario_tree(0.4, 0.0, float(u), belief, cost, controller.config).total_cost,
    )
    assert controller.name == "arcari_l0_passive_exploitation"
    assert controller.config.smpc_dual_horizon == 0
    assert action == float(expected)


def test_nominal_mpc_scalar_diagnostic_updates_belief_and_names_baseline():
    config, cost, belief = _small_problem()
    controller = NominalMPCScalar(belief.copy(), cost, config)
    action = controller.act(0.4, 0.0)
    before = controller.belief_mean
    controller.observe(0.4, 1.0, 0.4 + 1.2, config.process_var)

    assert controller.name == "nominal_mpc_ce"
    assert action in {float(u) for u in config.action_grid}
    assert controller.belief_mean != before


def test_arcari_cartpole_l_zero_diagnostic_controller_configures_l0():
    config, cost, dynamics, belief, state = _small_cartpole_problem()
    controller = ArcariPassiveExploitationCartPole(belief.copy(), dynamics, cost, config)
    action = controller.act(state, 0.0)
    tree = controller.build_tree_for_action(state, 0.0, action)

    assert controller.name == "arcari_l0_passive_exploitation"
    assert controller.config.smpc_dual_horizon == 0
    assert tree.dual_horizon == 0
    assert tree.node_count_by_depth() == [1]


def test_nominal_mpc_cartpole_diagnostic_updates_belief_and_time():
    config, cost, dynamics, belief, state = _small_cartpole_problem()
    controller = NominalMPCCartPole(belief.copy(), dynamics, cost, config)
    action = controller.act(state, 0.0)
    next_state, _ = cartpole_step(state, action, 1.1, dynamics)
    before = controller.belief_mean
    controller.observe(state, action, next_state)

    assert controller.name == "nominal_mpc_ce"
    assert action in {float(u) for u in config.action_grid}
    assert controller.t == 1
    assert controller.belief_mean != before


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

def test_arcari_scalar_continuous_action_can_be_non_grid_and_bounded():
    config = OfficialScalarConfig(
        horizon=1,
        action_low=-1.0,
        action_high=1.0,
        action_grid_size=3,
        continuous_actions=True,
        optimizer_grid_size=81,
        process_var=0.0,
        smpc_dual_horizon=1,
        smpc_scenarios=1,
    )
    cost = ScalarCost(ScalarCostConfig(state_weight=1.0, energy_weight=0.1, terminal_weight=1.0))
    controller = ArcariDualSMPCScalar(GaussianBelief(mean=1.0, var=0.0, process_var=0.0), cost, config)

    action = controller.act(0.4, 0.0)

    assert config.action_low <= action <= config.action_high
    assert action not in {float(u) for u in config.action_grid}
    assert math.isclose(action, -0.4 / 1.1, rel_tol=2e-3, abs_tol=2e-3)


def test_arcari_scalar_continuous_tree_keeps_inner_node_actions_continuous():
    config = OfficialScalarConfig(
        horizon=2,
        action_low=-1.0,
        action_high=1.0,
        action_grid_size=3,
        continuous_actions=True,
        optimizer_grid_size=81,
        process_var=0.0,
        smpc_dual_horizon=2,
        smpc_scenarios=1,
    )
    cost = ScalarCost(ScalarCostConfig(state_weight=1.0, energy_weight=0.1, terminal_weight=1.0))
    tree = build_arcari_scalar_scenario_tree(0.4, 0.0, -0.3, GaussianBelief(mean=1.0, var=0.0, process_var=0.0), cost, config)

    assert tree.node_count_by_depth() == [1, 1, 1]
    inner = tree.nodes[tree.levels[1][0]]
    assert inner.action is not None
    assert config.action_low <= inner.action <= config.action_high
    assert inner.action not in {float(u) for u in config.action_grid}


def test_diagnostic_arcari_scalar_forces_l_zero_and_nominal_mpc_is_ce():
    config, cost, belief = _small_problem()
    diag = ArcariPassiveExploitationScalar(belief.copy(), cost, config, seed=0)
    assert diag.name == "arcari_l0_passive_exploitation"
    assert diag.config.smpc_dual_horizon == 0
    tree = diag.build_tree_for_action(0.4, 0.0, 1.0)
    assert tree.dual_horizon == 0
    assert tree.node_count_by_depth() == [1]

    nominal = NominalMPCScalar(belief.copy(), cost, config)
    assert nominal.name == "nominal_mpc_ce"
    assert float(nominal.act(0.4, 0.0)) in {float(u) for u in config.action_grid}


def test_arcari_cartpole_continuous_action_can_be_non_grid_and_bounded(monkeypatch):
    config, cost, dynamics, belief, state = _small_cartpole_problem()
    config = OfficialCartPoleConfig(
        horizon=1,
        action_grid_size=3,
        continuous_actions=True,
        optimizer_grid_size=81,
        smpc_dual_horizon=1,
        smpc_scenarios=1,
        smpc_noise_std=(0.0, 0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        "controllers.official.build_arcari_cartpole_scenario_tree",
        lambda state, prev_action, t, root_action, belief, dynamics, cost, config: type(
            "Tree", (), {"total_cost": (float(root_action) - 0.37) ** 2, "root_action": float(root_action)}
        )(),
    )
    controller = ArcariDualSMPCCartPole(belief.copy(), dynamics, cost, config, seed=0)

    action = controller.act(state, 0.0)

    assert -1.0 <= action <= 1.0
    assert action not in {float(u) for u in config.action_grid}
    assert math.isclose(action, 0.37, rel_tol=2e-3, abs_tol=2e-3)


def test_diagnostic_arcari_cartpole_forces_l_zero_and_nominal_mpc_is_ce():
    config, cost, dynamics, belief, state = _small_cartpole_problem()
    diag = ArcariPassiveExploitationCartPole(belief.copy(), dynamics, cost, config, seed=0)
    assert diag.name == "arcari_l0_passive_exploitation"
    assert diag.config.smpc_dual_horizon == 0
    tree = diag.build_tree_for_action(state, 0.0, 0.5)
    assert tree.dual_horizon == 0
    assert tree.node_count_by_depth() == [1]

    nominal = NominalMPCCartPole(belief.copy(), dynamics, cost, config)
    assert nominal.name == "nominal_mpc_ce"
    assert float(nominal.act(state, 0.0)) in {float(u) for u in config.action_grid}

