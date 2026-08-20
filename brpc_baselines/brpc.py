"""Fixed-support BRPC-F calibration.

Implements the lightweight version specified in Section 8: theta particles,
particle-specific discrepancy inducing means, shared inducing covariance, log-domain
weights, coupled resampling, and Cholesky/solve based linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np


class TwinModel(Protocol):
    output_dim: int

    def batch_step(self, inputs: np.ndarray, theta: np.ndarray | float) -> np.ndarray: ...


@dataclass(frozen=True)
class BRPCConfig:
    theta_low: float = 0.0
    theta_high: float = 1.0
    num_particles: int = 128
    ess_fraction: float = 0.50
    eta_theta: float = 1.0
    eta_delta: float = 1.0
    rho_theta: float = 0.995
    rho_delta: float = 0.995
    theta_process_std: float = 0.01
    sigma_theta: float = 0.10
    sigma_epsilon: float = 0.03
    kernel_output_scale: float = 0.10
    kernel_length_scale: float | tuple[float, ...] = 0.25
    covariance_jitter: float = 1.0e-6
    covariance_inflation: float = 1.0e-5
    random_seed: int | None = None


@dataclass
class BRPCState:
    theta_particles: np.ndarray  # (N, theta_dim) currently theta_dim=1
    theta_weights: np.ndarray  # (N,)
    discrepancy_means: np.ndarray  # (N, output_dim, Mz)
    discrepancy_covariances: np.ndarray  # (output_dim, Mz, Mz)
    inducing_points: np.ndarray  # (Mz, input_dim)
    kzz: np.ndarray  # (Mz, Mz)
    time: int = 0

    def copy(self) -> "BRPCState":
        return BRPCState(
            theta_particles=self.theta_particles.copy(),
            theta_weights=self.theta_weights.copy(),
            discrepancy_means=self.discrepancy_means.copy(),
            discrepancy_covariances=self.discrepancy_covariances.copy(),
            inducing_points=self.inducing_points.copy(),
            kzz=self.kzz.copy(),
            time=self.time,
        )


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    m = float(np.max(values))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(values - m))))


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    return np.exp(log_weights - logsumexp(log_weights))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights * weights))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def cholesky_solve(a: np.ndarray, b: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    eye = np.eye(a.shape[0])
    last_error: np.linalg.LinAlgError | None = None
    for scale in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        try:
            L = np.linalg.cholesky(a + scale * jitter * eye)
            y = np.linalg.solve(L, b)
            return np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError as exc:
            last_error = exc
    raise np.linalg.LinAlgError(f"Cholesky failed after jitter escalation: {last_error}")


def log_normal_diag(y: np.ndarray, mean: np.ndarray, variance: float | np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    var = np.asarray(variance, dtype=float) + 0.0
    if var.ndim == 0:
        var = np.full_like(y, float(var))
    var = np.maximum(var.reshape(-1), 1e-12)
    diff = y - mean
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * var) + diff * diff / var))


def log_normal_full(y: np.ndarray, mean: np.ndarray, cov: np.ndarray, jitter: float) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    eye = np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov + jitter * eye)
    diff = y - mean
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, diff))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * (len(y) * np.log(2.0 * np.pi) + logdet + diff @ alpha))


class FixedSupportBRPC:
    def __init__(self, twin: TwinModel, inducing_points: np.ndarray, config: BRPCConfig = BRPCConfig()):
        self.twin = twin
        self.inducing_points = np.atleast_2d(np.asarray(inducing_points, dtype=float))
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.output_dim = int(getattr(twin, "output_dim", 1))
        self.state = self._initial_state()

    def _length_scales(self, input_dim: int) -> np.ndarray:
        ls = np.asarray(self.config.kernel_length_scale, dtype=float)
        if ls.ndim == 0:
            return np.full(input_dim, float(ls))
        if len(ls) != input_dim:
            raise ValueError("kernel_length_scale dimension mismatch.")
        return ls

    def kernel(self, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(x, dtype=float))
        XP = np.atleast_2d(np.asarray(xp, dtype=float))
        ls = self._length_scales(X.shape[1])
        diff = (X[:, None, :] - XP[None, :, :]) / ls
        sqdist = np.sum(diff * diff, axis=2)
        return self.config.kernel_output_scale**2 * np.exp(-0.5 * sqdist)

    def _initial_state(self) -> BRPCState:
        n = self.config.num_particles
        theta = self.rng.uniform(self.config.theta_low, self.config.theta_high, size=(n, 1))
        weights = np.full(n, 1.0 / n, dtype=float)
        m_z = self.inducing_points.shape[0]
        kzz = self.kernel(self.inducing_points, self.inducing_points)
        kzz = self._symmetrize(kzz + self.config.covariance_jitter * np.eye(m_z))
        means = np.zeros((n, self.output_dim, m_z), dtype=float)
        covs = np.stack([kzz.copy() for _ in range(self.output_dim)], axis=0)
        return BRPCState(theta, weights, means, covs, self.inducing_points.copy(), kzz)

    def reset_to_restart_prior(self) -> None:
        self.state = self._initial_state()

    def clone(self) -> "FixedSupportBRPC":
        other = FixedSupportBRPC(self.twin, self.inducing_points.copy(), replace(self.config, random_seed=None))
        other.rng = np.random.default_rng()
        other.state = self.state.copy()
        return other

    @staticmethod
    def _symmetrize(mat: np.ndarray) -> np.ndarray:
        return 0.5 * (mat + mat.T)

    def _clip_theta(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, self.config.theta_low, self.config.theta_high)

    def propagate(self, state: BRPCState | None = None, rng: np.random.Generator | None = None) -> BRPCState:
        src = self.state if state is None else state
        rng = self.rng if rng is None else rng
        cfg = self.config
        mid = 0.5 * (cfg.theta_low + cfg.theta_high)
        theta = mid + cfg.rho_theta * (src.theta_particles - mid)
        theta += rng.normal(0.0, cfg.theta_process_std, size=src.theta_particles.shape)
        theta = self._clip_theta(theta)
        means = cfg.rho_delta * src.discrepancy_means
        covs = []
        for j in range(self.output_dim):
            P = cfg.rho_delta**2 * src.discrepancy_covariances[j]
            P += (1.0 - cfg.rho_delta**2) * src.kzz
            P += cfg.covariance_inflation * np.eye(src.kzz.shape[0])
            covs.append(self._symmetrize(P))
        return BRPCState(theta, src.theta_weights.copy(), means, np.stack(covs), src.inducing_points.copy(), src.kzz.copy(), src.time)

    def support_map(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        kxz = self.kernel(X, self.state.inducing_points)
        kxx = self.kernel(X, X)
        kzz_inv_kzx = cholesky_solve(self.state.kzz, kxz.T, self.config.covariance_jitter)
        G = kzz_inv_kzx.T
        qf = self._symmetrize(kxx - kxz @ kzz_inv_kzx)
        return G, qf

    def _theta_log_likelihoods(self, predicted: BRPCState, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        y = np.atleast_2d(np.asarray(outputs, dtype=float))
        if y.shape[0] != np.atleast_2d(inputs).shape[0]:
            y = y.reshape(np.atleast_2d(inputs).shape[0], -1)
        logs = np.empty(len(predicted.theta_particles), dtype=float)
        var = self.config.sigma_theta**2
        for i, theta in enumerate(predicted.theta_particles):
            mean = self.twin.batch_step(inputs, theta)
            logs[i] = log_normal_diag(y, mean, var)
        return logs

    def update(self, inputs: np.ndarray, outputs: np.ndarray) -> BRPCState:
        return self.assimilate_from_state(self.propagate(), inputs, outputs)

    def assimilate_from_state(self, predicted: BRPCState, inputs: np.ndarray, outputs: np.ndarray) -> BRPCState:
        """Assimilate one observation batch from an already propagated state.

        BOCPD uses this to enforce prequential evidence ordering and avoid sampling
        the same transition twice in one decision step.
        """
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        Y = np.atleast_2d(np.asarray(outputs, dtype=float))
        if Y.shape[0] != X.shape[0]:
            Y = Y.reshape(X.shape[0], -1)

        logw = np.log(np.maximum(predicted.theta_weights, 1e-300))
        logw += self.config.eta_theta * self._theta_log_likelihoods(predicted, X, Y)
        weights = normalize_log_weights(logw)

        G, qf = self.support_map(X)
        m_z = predicted.kzz.shape[0]
        new_means = np.empty_like(predicted.discrepancy_means)
        new_covs = np.empty_like(predicted.discrepancy_covariances)
        R = qf + (self.config.sigma_epsilon**2 + self.config.covariance_jitter) * np.eye(X.shape[0])
        R = self._symmetrize(R)

        for j in range(self.output_dim):
            P = predicted.discrepancy_covariances[j]
            # Build information matrix with solves rather than explicit inverse in the update equations.
            pinv_eye = cholesky_solve(P, np.eye(m_z), self.config.covariance_jitter)
            rinv_G = cholesky_solve(R, G, self.config.covariance_jitter)
            info = pinv_eye + self.config.eta_delta * G.T @ rinv_G
            C = cholesky_solve(info, np.eye(m_z), self.config.covariance_jitter)
            C = self._symmetrize(C)
            new_covs[j] = C
            for i, theta in enumerate(predicted.theta_particles):
                twin_mean = self.twin.batch_step(X, theta)[:, j]
                residual = Y[:, j] - twin_mean
                prior_mean = predicted.discrepancy_means[i, j]
                rhs = pinv_eye @ prior_mean
                rhs += self.config.eta_delta * G.T @ cholesky_solve(R, residual, self.config.covariance_jitter)
                new_means[i, j] = C @ rhs

        updated = BRPCState(
            theta_particles=predicted.theta_particles,
            theta_weights=weights,
            discrepancy_means=new_means,
            discrepancy_covariances=new_covs,
            inducing_points=predicted.inducing_points,
            kzz=predicted.kzz,
            time=predicted.time + 1,
        )
        if effective_sample_size(weights) < self.config.ess_fraction * len(weights):
            idx = systematic_resample(weights, self.rng)
            updated.theta_particles = updated.theta_particles[idx]
            updated.discrepancy_means = updated.discrepancy_means[idx]
            updated.theta_weights = np.full(len(weights), 1.0 / len(weights), dtype=float)
        self.state = updated
        return self.state

    def predict_particle(self, inputs: np.ndarray, particle_index: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        G, qf = self.support_map(X)
        idx = int(particle_index)
        mean = self.twin.batch_step(X, self.state.theta_particles[idx]).copy()
        covs = []
        for j in range(self.output_dim):
            mean[:, j] += G @ self.state.discrepancy_means[idx, j]
            cov = qf + G @ self.state.discrepancy_covariances[j] @ G.T
            cov += self.config.sigma_epsilon**2 * np.eye(X.shape[0])
            covs.append(self._symmetrize(cov))
        return mean, np.stack(covs, axis=0)

    def predict(self, inputs: np.ndarray) -> dict:
        X = np.atleast_2d(np.asarray(inputs, dtype=float))
        means = []
        covs = []
        for i in range(len(self.state.theta_particles)):
            mean, cov = self.predict_particle(X, i)
            means.append(mean)
            covs.append(cov)
        return {
            "weights": self.state.theta_weights.copy(),
            "means": np.asarray(means),  # (N, batch, output_dim)
            "covariances": np.asarray(covs),  # (N, output_dim, batch, batch)
        }

    def predictive_mean(self, inputs: np.ndarray) -> np.ndarray:
        pred = self.predict(inputs)
        return np.tensordot(pred["weights"], pred["means"], axes=(0, 0))

    def log_predictive(self, inputs: np.ndarray, outputs: np.ndarray, propagated: BRPCState | None = None) -> float:
        old_state = self.state
        if propagated is not None:
            self.state = propagated
        try:
            X = np.atleast_2d(np.asarray(inputs, dtype=float))
            Y = np.atleast_2d(np.asarray(outputs, dtype=float))
            if Y.shape[0] != X.shape[0]:
                Y = Y.reshape(X.shape[0], -1)
            G, qf = self.support_map(X)
            components = []
            for i, theta in enumerate(self.state.theta_particles):
                logp = np.log(max(self.state.theta_weights[i], 1e-300))
                mean = self.twin.batch_step(X, theta)
                for j in range(self.output_dim):
                    mu = mean[:, j] + G @ self.state.discrepancy_means[i, j]
                    cov = qf + G @ self.state.discrepancy_covariances[j] @ G.T
                    cov += self.config.sigma_epsilon**2 * np.eye(X.shape[0])
                    logp += log_normal_full(Y[:, j], mu, cov, self.config.covariance_jitter)
                components.append(logp)
            return logsumexp(np.asarray(components))
        finally:
            self.state = old_state

    def sample_latent(self, rng: np.random.Generator | None = None, state: BRPCState | None = None) -> dict:
        rng = self.rng if rng is None else rng
        src = self.state if state is None else state
        i = int(rng.choice(len(src.theta_weights), p=src.theta_weights))
        u = []
        for j in range(self.output_dim):
            u.append(rng.multivariate_normal(src.discrepancy_means[i, j], src.discrepancy_covariances[j]))
        return {"particle_index": i, "theta": src.theta_particles[i].copy(), "u": np.asarray(u)}

    def sample_latent_path(self, horizon: int, rng: np.random.Generator | None = None) -> list[dict]:
        """Sample a coherent future theta/discrepancy path from the BRPC dynamics.

        The path first selects one posterior particle, then evolves that particle's
        theta and inducing discrepancy values with the same transition model used by
        BRPC filtering. The live filter state is left unchanged.
        """

        rng = self.rng if rng is None else rng
        horizon = int(horizon)
        if horizon <= 0:
            return []
        cfg = self.config
        particle_index = int(rng.choice(len(self.state.theta_weights), p=self.state.theta_weights))
        mid = 0.5 * (cfg.theta_low + cfg.theta_high)
        theta = self.state.theta_particles[particle_index].copy()
        u = []
        for j in range(self.output_dim):
            u.append(rng.multivariate_normal(self.state.discrepancy_means[particle_index, j], self.state.discrepancy_covariances[j]))
        u = np.asarray(u, dtype=float)

        path = []
        for _ in range(horizon):
            theta = mid + cfg.rho_theta * (theta - mid)
            theta += rng.normal(0.0, cfg.theta_process_std, size=theta.shape)
            theta = self._clip_theta(theta)
            for j in range(self.output_dim):
                base_kzz = self.state.kzz.copy()
                if cfg.covariance_jitter > 0.0 and np.all(np.diag(base_kzz) >= cfg.covariance_jitter):
                    base_kzz = base_kzz - cfg.covariance_jitter * np.eye(self.state.kzz.shape[0])
                innovation_cov = (1.0 - cfg.rho_delta**2) * base_kzz
                innovation_cov += cfg.covariance_inflation * np.eye(self.state.kzz.shape[0])
                innovation_cov = self._symmetrize(innovation_cov)
                eigvals, eigvecs = np.linalg.eigh(innovation_cov)
                eigvals = np.maximum(eigvals, 0.0)
                innovation_cov = self._symmetrize((eigvecs * eigvals) @ eigvecs.T)
                if np.max(np.abs(innovation_cov)) > 0.0:
                    innovation = rng.multivariate_normal(np.zeros(self.state.kzz.shape[0]), innovation_cov)
                else:
                    innovation = np.zeros(self.state.kzz.shape[0])
                u[j] = cfg.rho_delta * u[j] + innovation
            path.append({"particle_index": particle_index, "theta": theta.copy(), "u": u.copy()})
        return path

    def diagnostics(self) -> dict:
        w = self.state.theta_weights
        return {
            "time": self.state.time,
            "theta_mean": float(np.sum(w * self.state.theta_particles[:, 0])),
            "theta_var": float(np.sum(w * (self.state.theta_particles[:, 0] - np.sum(w * self.state.theta_particles[:, 0])) ** 2)),
            "ess": effective_sample_size(w),
            "num_particles": len(w),
            "weights_sum": float(np.sum(w)),
        }
