"""Lightweight BOCPD-BRPC expert mixture.

This implements the spec's prequential evidence ordering, fresh restart expert,
log-domain expert masses, hard-anchor rule, pruning, and one assimilation per
retained expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brpc import BRPCState, FixedSupportBRPC, logsumexp, normalize_log_weights


@dataclass(frozen=True)
class BOCPDConfig:
    hazard: float = 0.01
    max_experts: int = 10
    restart_margin_rho_B: float = 1.0
    min_segment_length: int = 10


@dataclass
class Expert:
    start_time: int
    log_mass: float
    brpc: FixedSupportBRPC

    @property
    def mass(self) -> float:
        return float(np.exp(self.log_mass))


class BOCPDBRPC:
    def __init__(self, anchor_brpc: FixedSupportBRPC, config: BOCPDConfig = BOCPDConfig()):
        self.config = config
        self.experts = [Expert(start_time=0, log_mass=0.0, brpc=anchor_brpc.clone())]
        self.anchor_start_time = 0
        self.time = 0
        self.restart_event = False

    def _restart_prior_expert(self, start_time: int) -> Expert:
        brpc = self.experts[0].brpc.clone()
        brpc.reset_to_restart_prior()
        return Expert(start_time=start_time, log_mass=-np.inf, brpc=brpc)

    def _current_hazard(self) -> float:
        anchor_age = self.time - self.anchor_start_time
        if anchor_age < self.config.min_segment_length:
            return 0.0
        return float(np.clip(self.config.hazard, 0.0, 1.0))

    def update(self, inputs: np.ndarray, outputs: np.ndarray) -> list[Expert]:
        hazard = self._current_hazard()
        candidates: list[tuple[Expert, BRPCState, float, bool]] = []

        for expert in self.experts:
            propagated = expert.brpc.propagate()
            log_ev = expert.brpc.log_predictive(inputs, outputs, propagated=propagated)
            log_mass = np.log(max(1.0 - hazard, 1e-300)) + expert.log_mass + log_ev
            candidates.append((expert, propagated, log_mass, False))

        if hazard > 0.0:
            fresh = self._restart_prior_expert(self.time)
            fresh_prior = fresh.brpc.state.copy()
            log_ev = fresh.brpc.log_predictive(inputs, outputs, propagated=fresh_prior)
            candidates.append((fresh, fresh_prior, np.log(hazard) + log_ev, True))

        log_norm = logsumexp(np.asarray([c[2] for c in candidates], dtype=float))
        candidates = [(e, s, lm - log_norm, is_fresh) for (e, s, lm, is_fresh) in candidates]

        self.restart_event = False
        anchor = self._find_anchor(candidates)
        post_anchor = [c for c in candidates if c[0].start_time > self.anchor_start_time]
        if anchor is not None and post_anchor:
            best = max(post_anchor, key=lambda c: c[2])
            if np.exp(best[2]) > self.config.restart_margin_rho_B * np.exp(anchor[2]):
                self.anchor_start_time = best[0].start_time
                self.restart_event = True
                candidates = [c for c in candidates if c[0].start_time >= self.anchor_start_time]

        candidates = self._prune(candidates)
        renorm = logsumexp(np.asarray([c[2] for c in candidates], dtype=float))
        retained: list[Expert] = []
        for expert, propagated, log_mass, _ in candidates:
            expert.brpc.assimilate_from_state(propagated, inputs, outputs)
            expert.log_mass = float(log_mass - renorm)
            retained.append(expert)
        self.experts = retained
        self.time += 1
        return self.experts

    def _find_anchor(self, candidates: list[tuple[Expert, BRPCState, float, bool]]) -> tuple[Expert, BRPCState, float, bool] | None:
        for c in candidates:
            if c[0].start_time == self.anchor_start_time:
                return c
        return None

    def _prune(self, candidates: list[tuple[Expert, BRPCState, float, bool]]) -> list[tuple[Expert, BRPCState, float, bool]]:
        if len(candidates) <= self.config.max_experts:
            return candidates
        must_keep = set()
        anchor_indices = [idx for idx, c in enumerate(candidates) if c[0].start_time == self.anchor_start_time]
        if anchor_indices:
            must_keep.add(max(anchor_indices, key=lambda idx: candidates[idx][2]))
        for idx, c in enumerate(candidates):
            if c[3]:
                must_keep.add(idx)
        remaining_slots = max(0, self.config.max_experts - len(must_keep))
        optional = [idx for idx in range(len(candidates)) if idx not in must_keep]
        optional.sort(key=lambda idx: candidates[idx][2], reverse=True)
        keep = must_keep.union(optional[:remaining_slots])
        return [candidates[idx] for idx in sorted(keep, key=lambda idx: candidates[idx][2], reverse=True)]

    def predict(self, inputs: np.ndarray) -> dict:
        weights = []
        means = []
        for expert in self.experts:
            pred = expert.brpc.predict(inputs)
            for w, m in zip(pred["weights"], pred["means"]):
                weights.append(expert.mass * w)
                means.append(m)
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
        return {"weights": weights, "means": np.asarray(means)}

    def predictive_mean(self, inputs: np.ndarray) -> np.ndarray:
        pred = self.predict(inputs)
        return np.tensordot(pred["weights"], pred["means"], axes=(0, 0))

    def sample_latent(self, rng: np.random.Generator | None = None) -> dict:
        rng = np.random.default_rng() if rng is None else rng
        expert = self._sample_expert(rng)
        sample = expert.brpc.sample_latent(rng)
        sample["expert_start_time"] = expert.start_time
        return sample

    def _sample_expert(self, rng: np.random.Generator) -> Expert:
        masses = np.asarray([e.mass for e in self.experts], dtype=float)
        masses = masses / np.sum(masses)
        e_idx = int(rng.choice(len(self.experts), p=masses))
        return self.experts[e_idx]

    def sample_latent_path(self, horizon: int, rng: np.random.Generator | None = None) -> list[dict]:
        """Sample a coherent fixed-expert future path for BOCPD-BRPC planning.

        This planner variant samples the BOCPD expert once, then propagates and samples
        a BRPC latent path inside that expert over the planning horizon. It deliberately
        does not sample future BOCPD changepoints/restarts, so PS-BBRPC denotes
        fixed-expert Thompson MPC rather than hazard-rollout Thompson MPC.
        """

        rng = np.random.default_rng() if rng is None else rng
        expert = self._sample_expert(rng)
        path = expert.brpc.sample_latent_path(horizon, rng)
        for sample in path:
            sample["expert_start_time"] = expert.start_time
        return path

    def diagnostics(self) -> dict:
        masses = np.asarray([e.mass for e in self.experts], dtype=float)
        starts = [e.start_time for e in self.experts]
        recent = float(sum(e.mass for e in self.experts if self.time - e.start_time <= self.config.min_segment_length))
        return {
            "expert_masses": masses,
            "expert_start_times": starts,
            "expert_mass_sum": float(np.sum(masses)),
            "anchor_start_time": self.anchor_start_time,
            "restart_event": self.restart_event,
            "recent_change_probability": recent,
            "num_experts": len(self.experts),
            "inner_weight_sums": [float(np.sum(e.brpc.state.theta_weights)) for e in self.experts],
        }
