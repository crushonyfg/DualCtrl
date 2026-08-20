import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import brpc_baselines.brpc as brpc_module
from brpc_baselines.bocpd_brpc import BOCPDBRPC, BOCPDConfig
from brpc_baselines.brpc import BRPCConfig, FixedSupportBRPC
from brpc_baselines.geometry import generate_geometry_csv, toy2_diagnostic_conditions, toy2_geometry_rows
from brpc_baselines.planners import CEPlanner, CEMConfig, PosteriorSamplingPlanner
from brpc_baselines.smoke_runner import BASELINE_MATRIX, run_matrix
from brpc_baselines.toy_envs import Toy1Config, Toy1DigitalTwin, Toy1PhysicalEnv, Toy2Config, Toy2DigitalTwin, Toy2PhysicalEnv


def test_toy1_dynamics_and_reward_accounting_without_noise():
    cfg = Toy1Config(
        horizon_T=4,
        theta_initial=0.85,
        beta_initial=0.08,
        kappa_delta=0.03,
        sigma_w=0.0,
        q_x=1.0,
        lambda_energy=0.05,
        lambda_switch=0.20,
        x0=0.25,
    )
    env = Toy1PhysicalEnv(cfg, noise_path=np.zeros(cfg.horizon_T))
    state0 = env.reset()
    action = np.array([0.4])
    next_state, reward, done, info = env.step(action)

    expected_next = 0.85 * state0[0] + 0.4 + 0.08 * np.tanh(2.0 * state0[0]) + 0.03 * 0.4 * abs(0.4)
    assert np.allclose(next_state, [expected_next])
    assert np.allclose(info["calibration_input"], [state0[0], 0.4])
    assert np.allclose(info["calibration_output"], next_state)
    assert reward.task_reward == -cfg.q_x * (state0[0] - 0.0) ** 2
    assert reward.energy_cost == cfg.lambda_energy * 0.4**2
    assert reward.switching_cost == cfg.lambda_switch * 0.4**2
    assert reward.net_reward == reward.task_reward - reward.energy_cost - reward.switching_cost
    assert not done


def test_toy2_response_reward_and_state_is_previous_action():
    cfg = Toy2Config(horizon_T=3, theta_initial=0.1, sigma_y=0.0, discrepancy_sine_amplitude=0.05)
    env = Toy2PhysicalEnv(cfg, noise_path=np.zeros(cfg.horizon_T))
    state0 = env.reset()
    assert np.allclose(state0, [cfg.a_left])

    action = np.array([cfg.a_diag])
    next_state, reward, done, info = env.step(action)
    twin_response = env.twin.step(state0, action, cfg.theta_initial)[0]
    expected_y = twin_response + cfg.discrepancy_sine_amplitude * np.sin(4.0 * np.pi * cfg.a_diag)
    expected_energy = cfg.lambda_energy * cfg.a_diag**2
    expected_switch = cfg.lambda_switch * (cfg.a_diag - cfg.a_left) ** 2

    assert np.allclose(next_state, [cfg.a_diag])
    assert np.allclose(info["calibration_input"], [cfg.a_left, cfg.a_diag])
    assert np.allclose(info["calibration_output"], [expected_y])
    assert reward.task_reward == expected_y
    assert reward.energy_cost == expected_energy
    assert reward.switching_cost == expected_switch
    assert reward.net_reward == expected_y - expected_energy - expected_switch
    assert not done


def test_toy2_diagnostic_action_conditions_pass_geometry_gate():
    cfg = Toy2Config()
    conditions = toy2_diagnostic_conditions(cfg)

    assert conditions.a_diag_not_production_optimal
    assert conditions.production_gap_at_diag > 0.0
    assert conditions.information_higher_at_a_diag
    assert conditions.diagnostic_information > conditions.left_information
    assert conditions.diagnostic_information > conditions.right_information
    assert conditions.bridge_switching_inequality
    assert conditions.bridge_cost < conditions.direct_cost
    assert conditions.passes


def test_toy2_geometry_rows_include_required_screening_outputs():
    cfg = Toy2Config()
    rows = toy2_geometry_rows(cfg, num_state=3, num_action=5)
    assert len(rows) == 15
    required = {
        "expected_reward_old",
        "expected_reward_new",
        "parameter_sensitivity",
        "fisher_proxy",
        "variance_reduction_proxy",
        "predictive_kl_old_new",
        "switching_cost",
    }
    assert required <= set(rows[0])
    diag_row = min(rows, key=lambda r: abs(float(r["action"]) - cfg.a_diag))
    left_row = min(rows, key=lambda r: abs(float(r["action"]) - cfg.a_left))
    assert float(diag_row["predictive_kl_old_new"]) > float(left_row["predictive_kl_old_new"])
    assert float(diag_row["parameter_sensitivity"]) > float(left_row["parameter_sensitivity"])


def test_generate_geometry_screening_csv_cli_backend(tmp_path):
    out = tmp_path / "toy2_geometry.csv"
    generate_geometry_csv("toy2", out, num_state=2, num_action=3)
    text = out.read_text()
    assert "expected_reward_old" in text
    assert "fisher_proxy" in text
    assert "predictive_kl_old_new" in text
    assert len(text.strip().splitlines()) == 1 + 2 * 3


def test_brpc_update_shapes_weight_normalization_and_covariance_psd():
    cfg = Toy2Config(horizon_T=5, sigma_y=0.01)
    twin = Toy2DigitalTwin(cfg)
    inducing = np.column_stack([np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8)])
    brpc = FixedSupportBRPC(
        twin,
        inducing,
        BRPCConfig(
            theta_low=0.0,
            theta_high=1.0,
            num_particles=16,
            sigma_theta=0.2,
            sigma_epsilon=0.01,
            kernel_output_scale=0.05,
            kernel_length_scale=(0.4, 0.4),
            random_seed=123,
        ),
    )
    X = np.array([[0.2, 0.2], [0.2, 0.5], [0.5, 0.8]])
    Y = twin.batch_step(X, 0.1)
    state = brpc.update(X, Y)

    assert state.theta_particles.shape == (16, 1)
    assert state.theta_weights.shape == (16,)
    assert np.isclose(np.sum(state.theta_weights), 1.0)
    assert state.discrepancy_means.shape == (16, 1, 8)
    assert state.discrepancy_covariances.shape == (1, 8, 8)
    assert np.allclose(state.discrepancy_covariances[0], state.discrepancy_covariances[0].T)
    assert np.min(np.linalg.eigvalsh(state.discrepancy_covariances[0])) >= -1e-10
    pred = brpc.predict(X[:2])
    assert pred["weights"].shape == (16,)
    assert pred["means"].shape == (16, 2, 1)
    assert pred["covariances"].shape == (16, 1, 2, 2)
    assert np.allclose(pred["covariances"][0, 0], pred["covariances"][0, 0].T)
    assert np.min(np.linalg.eigvalsh(pred["covariances"][0, 0])) >= -1e-10


def test_particle_and_expert_weights_sum_to_one_after_updates():
    cfg = Toy2Config(horizon_T=6, sigma_y=0.0, change_time=3)
    env = Toy2PhysicalEnv(cfg, seed=7)
    twin = Toy2DigitalTwin(cfg)
    inducing = np.column_stack([np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)])
    brpc = FixedSupportBRPC(
        twin,
        inducing,
        BRPCConfig(
            num_particles=12,
            ess_fraction=0.0,
            sigma_theta=0.15,
            sigma_epsilon=0.01,
            kernel_output_scale=0.04,
            kernel_length_scale=(0.35, 0.35),
            random_seed=4,
        ),
    )
    bbrpc = BOCPDBRPC(brpc, BOCPDConfig(hazard=0.30, max_experts=4, min_segment_length=1))

    state = env.reset()
    for t, action in enumerate([cfg.a_left, cfg.a_diag, cfg.a_right, cfg.a_diag, cfg.a_left]):
        del t, state
        state, _, _, info = env.step(np.array([action]))
        experts = bbrpc.update(info["calibration_input"][None, :], info["calibration_output"][None, :])
        expert_masses = np.array([expert.mass for expert in experts])
        assert np.isclose(np.sum(expert_masses), 1.0)
        for expert in experts:
            assert np.isclose(np.sum(expert.brpc.state.theta_weights), 1.0)


class ScriptedBOCPDState:
    def __init__(self, mean=0.0, observations_seen=0):
        self.mean = float(mean)
        self.observations_seen = int(observations_seen)

    def copy(self):
        return ScriptedBOCPDState(self.mean, self.observations_seen)


class ScriptedBOCPDBRPC:
    def __init__(self, shared=None, state=None):
        self.shared = {"next_id": 0, "events": []} if shared is None else shared
        self.object_id = self.shared["next_id"]
        self.shared["next_id"] += 1
        self.state = ScriptedBOCPDState() if state is None else state
        self.assimilate_calls = 0

    def clone(self):
        return ScriptedBOCPDBRPC(self.shared, self.state.copy())

    def reset_to_restart_prior(self):
        self.state = ScriptedBOCPDState()
        self.shared["events"].append(("reset", self.object_id))

    def propagate(self):
        self.shared["events"].append(("propagate", self.object_id, self.state.observations_seen))
        return self.state.copy()

    def log_predictive(self, inputs, outputs, propagated=None):
        del inputs
        state = self.state if propagated is None else propagated
        y = float(np.asarray(outputs).reshape(-1)[0])
        # A never-assimilated restart prior is deliberately broad. Once an expert
        # has assimilated data, stale means are penalized sharply on a jump.
        log_ev = 0.0 if state.observations_seen == 0 else -abs(y - state.mean)
        self.shared["events"].append(("log_predictive", self.object_id, state.observations_seen, state.mean, y, log_ev))
        return log_ev

    def assimilate_from_state(self, propagated, inputs, outputs):
        del inputs
        y = float(np.asarray(outputs).reshape(-1)[0])
        self.shared["events"].append(("assimilate", self.object_id, propagated.observations_seen, y))
        self.assimilate_calls += 1
        self.state = propagated.copy()
        self.state.mean = y
        self.state.observations_seen += 1
        return self.state


def test_bocpd_uses_prequential_evidence_before_assimilation():
    shared = {"next_id": 0, "events": []}
    bbrpc = BOCPDBRPC(
        ScriptedBOCPDBRPC(shared),
        BOCPDConfig(hazard=0.25, max_experts=5, restart_margin_rho_B=100.0, min_segment_length=0),
    )

    bbrpc.update(np.array([[0.0]]), np.array([[1.0]]))

    first_assimilation = min(i for i, event in enumerate(shared["events"]) if event[0] == "assimilate")
    evidence_indices = [i for i, event in enumerate(shared["events"]) if event[0] == "log_predictive"]
    assert evidence_indices
    assert all(i < first_assimilation for i in evidence_indices)
    assert all(shared["events"][i][2] == 0 for i in evidence_indices)


def test_bocpd_assimilates_each_retained_expert_exactly_once_per_step():
    bbrpc = BOCPDBRPC(
        ScriptedBOCPDBRPC(),
        BOCPDConfig(hazard=0.25, max_experts=10, restart_margin_rho_B=100.0, min_segment_length=0),
    )

    for y in (0.0, 0.5, 1.0):
        before = {expert.brpc.object_id: expert.brpc.assimilate_calls for expert in bbrpc.experts}
        retained = bbrpc.update(np.array([[0.0]]), np.array([[y]]))
        for expert in retained:
            previous_calls = before.get(expert.brpc.object_id, 0)
            assert expert.brpc.assimilate_calls == previous_calls + 1


def test_bocpd_no_change_does_not_trigger_false_restart():
    bbrpc = BOCPDBRPC(
        ScriptedBOCPDBRPC(),
        BOCPDConfig(hazard=0.20, max_experts=8, restart_margin_rho_B=1.0, min_segment_length=0),
    )

    for _ in range(6):
        bbrpc.update(np.array([[0.0]]), np.array([[0.0]]))
        assert not bbrpc.restart_event
        assert bbrpc.anchor_start_time == 0


def test_bocpd_large_jump_increases_fresh_expert_mass():
    bbrpc = BOCPDBRPC(
        ScriptedBOCPDBRPC(),
        BOCPDConfig(hazard=0.20, max_experts=8, restart_margin_rho_B=100.0, min_segment_length=1),
    )
    bbrpc.update(np.array([[0.0]]), np.array([[0.0]]))
    stable_start_time = bbrpc.time
    bbrpc.update(np.array([[0.0]]), np.array([[0.0]]))
    stable_fresh_mass = next(expert.mass for expert in bbrpc.experts if expert.start_time == stable_start_time)

    jump_start_time = bbrpc.time
    bbrpc.update(np.array([[0.0]]), np.array([[10.0]]))
    jump_fresh_mass = next(expert.mass for expert in bbrpc.experts if expert.start_time == jump_start_time)

    assert jump_fresh_mass > stable_fresh_mass
    assert jump_fresh_mass > 0.5


def test_bocpd_pruning_retains_anchor_and_fresh_experts_and_normalizes_masses():
    bbrpc = BOCPDBRPC(
        ScriptedBOCPDBRPC(),
        BOCPDConfig(hazard=0.30, max_experts=2, restart_margin_rho_B=100.0, min_segment_length=1),
    )
    bbrpc.update(np.array([[0.0]]), np.array([[0.0]]))

    fresh_start_time = bbrpc.time
    experts = bbrpc.update(np.array([[0.0]]), np.array([[0.0]]))
    starts = {expert.start_time for expert in experts}
    masses = np.array([expert.mass for expert in experts])

    assert starts == {bbrpc.anchor_start_time, fresh_start_time}
    assert len(experts) == 2
    assert np.isclose(np.sum(masses), 1.0)
    assert all(np.isfinite(expert.log_mass) for expert in experts)
    assert all(np.isclose(np.sum(expert.brpc.state.theta_weights), 1.0) for expert in experts if hasattr(expert.brpc.state, "theta_weights"))


def test_coupled_resampling_preserves_theta_discrepancy_alignment():
    twin = Toy1DigitalTwin()
    inducing = np.column_stack([np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6)])
    brpc = FixedSupportBRPC(
        twin,
        inducing,
        BRPCConfig(
            theta_low=0.0,
            theta_high=1.0,
            num_particles=6,
            ess_fraction=1.1,
            sigma_theta=0.001,
            sigma_epsilon=0.02,
            eta_delta=0.0,
            rho_theta=1.0,
            rho_delta=1.0,
            kernel_output_scale=0.05,
            kernel_length_scale=(0.5, 0.5),
            theta_process_std=0.0,
            random_seed=10,
        ),
    )
    brpc.state.theta_particles[:, 0] = np.linspace(0.0, 1.0, 6)
    for idx in range(6):
        brpc.state.discrepancy_means[idx, 0, :] = 100.0 + idx

    X = np.array([[0.7, 0.2], [0.4, -0.1]])
    Y = twin.batch_step(X, brpc.state.theta_particles[4])
    original_resample = brpc_module.systematic_resample
    brpc_module.systematic_resample = lambda weights, rng: np.array([5, 4, 3, 2, 1, 0])
    try:
        state = brpc.update(X, Y)
    finally:
        brpc_module.systematic_resample = original_resample

    assert np.isclose(np.sum(state.theta_weights), 1.0)
    assert np.allclose(state.theta_particles[:, 0], [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    assert np.allclose(state.discrepancy_means[:, 0, 0], [105.0, 104.0, 103.0, 102.0, 101.0, 100.0])


class RecordingTwin:
    state_dim = 1
    action_dim = 1
    output_dim = 1

    def __init__(self):
        self.inputs = []

    def batch_step(self, inputs, theta):
        del theta
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        self.inputs.extend(X.copy())
        return (10.0 * X[:, 0] + X[:, 1])[:, None]


class FixedLatentCalibrator:
    def __init__(self, theta, u):
        self.theta = np.asarray(theta, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.sample_calls = 0
        self.predictive_mean_calls = 0

    def sample_latent(self, rng=None):
        del rng
        self.sample_calls += 1
        return {"particle_index": 0, "theta": self.theta.copy(), "u": self.u.copy()}

    def predictive_mean(self, inputs):
        self.predictive_mean_calls += 1
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        return np.zeros((X.shape[0], 1))


def test_ps_discrepancy_sample_is_coherent_over_rollout():
    twin = RecordingTwin()
    inducing = np.array([[0.0, -1.0], [0.0, 1.0]])
    latent_u = np.array([[0.25, 0.25]])
    calibrator = FixedLatentCalibrator(theta=np.array([0.3]), u=latent_u)
    planner = PosteriorSamplingPlanner(
        twin=twin,
        inducing_points=inducing,
        kernel_fn=lambda x, xp: np.ones((np.atleast_2d(x).shape[0], np.atleast_2d(xp).shape[0])),
        reward_fn=lambda state, action, previous_action, t: float(state[0]),
        config=CEMConfig(
            horizon=3,
            population=1,
            elite_fraction=1.0,
            iterations=1,
            smoothing=0.0,
            action_low=0.0,
            action_high=0.0,
            random_seed=1,
        ),
    )

    planner.act(np.array([1.0]), np.array([0.0]), calibrator, t=0)

    assert calibrator.sample_calls == 1
    assert calibrator.predictive_mean_calls == 0
    rolled_out_states = np.asarray(twin.inputs)[:, 0]
    assert np.allclose(rolled_out_states, [1.0, 10.25, 102.75])


def test_ce_and_ps_planners_use_only_their_allowed_information():
    ce_source = inspect.getsource(CEPlanner.act)
    ps_source = inspect.getsource(PosteriorSamplingPlanner.act)
    for forbidden in ("ucb", "UCB", "bonus", "dual"):
        assert forbidden not in ce_source
        assert forbidden not in ps_source

    class StrictCECalibrator:
        def __init__(self):
            self.predictive_mean_calls = 0

        def predictive_mean(self, inputs):
            self.predictive_mean_calls += 1
            X = np.atleast_2d(np.asarray(inputs, dtype=float))
            return X[:, :1]

        def sample_latent(self, rng=None):
            raise AssertionError("CE planner must not sample posterior latents")

    class StrictPSCalibrator(FixedLatentCalibrator):
        def predictive_mean(self, inputs):
            del inputs
            raise AssertionError("PS planner must not use certainty-equivalent mean")

    cem_cfg = CEMConfig(horizon=2, population=2, elite_fraction=0.5, iterations=1, action_low=0.0, action_high=0.0, random_seed=2)
    reward = lambda state, action, previous_action, t: 0.0
    ce_calibrator = StrictCECalibrator()
    ce = CEPlanner(reward, cem_cfg)
    ce.act(np.array([0.1]), np.array([0.0]), ce_calibrator, t=0)
    assert ce_calibrator.predictive_mean_calls == cem_cfg.horizon * cem_cfg.population * cem_cfg.iterations

    ps_calibrator = StrictPSCalibrator(theta=np.array([0.4]), u=np.zeros((1, 2)))
    ps = PosteriorSamplingPlanner(
        Toy1DigitalTwin(),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        lambda x, xp: np.eye(np.atleast_2d(x).shape[0], np.atleast_2d(xp).shape[0]),
        reward,
        cem_cfg,
    )
    ps.act(np.array([0.1]), np.array([0.0]), ps_calibrator, t=0)
    assert ps_calibrator.sample_calls == 1


def test_smoke_runner_executes_2x2_matrix():
    results = run_matrix(seed=5, horizon=3)
    assert [r.baseline for r in results] == list(BASELINE_MATRIX)
    assert all(r.steps == 3 for r in results)
    assert all(np.isfinite(r.total_return) for r in results)
    assert all(r.planner_queries > 0 for r in results)
