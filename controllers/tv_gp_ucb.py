"""Bogunovic, Scarlett, and Cevher (2016) time-varying GP-UCB.

This module implements the finite-action version of the time-varying Gaussian
process bandit model from "Time-Varying Gaussian Process Bandit Optimization".
The model is the Markov GP evolution

    f_{t+1}(x) = sqrt(1 - epsilon) f_t(x) + sqrt(epsilon) g_{t+1}(x),
    g_{t+1} ~ GP(0, k_x),

which implies the product time-space covariance

    k((x,t), (x',t')) = (1 - epsilon) ** (|t - t'| / 2) k_x(x, x').

TVGPUCB implements the paper's reward maximization acquisition. TVGPLCB is the
sign-flipped cost-minimization counterpart: it uses the same posterior and
confidence width, but selects the arm with the smallest lower confidence bound.
RGP-UCB is included only as an auxiliary comparison baseline from the paper; it
resets a stationary finite-action GP-UCB instance every batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

ArmT = TypeVar("ArmT")
ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class TVGPUCBConfig:
    """Configuration for finite-action TV-GP-UCB/LCB.

    Attributes:
        epsilon: Markov innovation probability in the paper's time-varying GP.
        noise_var: Observation-noise variance, i.e. sigma^2 in y_t=f_t(x_t)+z_t.
        delta: Failure probability in the finite-action confidence parameter.
        lengthscale: Squared-exponential spatial-kernel lengthscale.
        signal_var: Spatial-kernel marginal variance k_x(x, x).
        jitter: Numerical diagonal jitter added in addition to noise_var.
    """

    epsilon: float = 0.02
    noise_var: float = 1e-3
    delta: float = 0.1
    lengthscale: float = 1.0
    signal_var: float = 1.0
    jitter: float = 1e-10

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if self.noise_var < 0.0:
            raise ValueError("noise_var must be nonnegative")
        if not (0.0 < self.delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive")
        if self.signal_var <= 0.0:
            raise ValueError("signal_var must be positive")
        if self.jitter < 0.0:
            raise ValueError("jitter must be nonnegative")


def squared_exponential_kernel(x: np.ndarray, y: np.ndarray, lengthscale: float = 1.0, signal_var: float = 1.0) -> float:
    """Spatial kernel k_x(x, y) used in the finite-action implementation."""

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"kernel inputs must have the same shape, got {x.shape} and {y.shape}")
    diff = x - y
    return float(signal_var * np.exp(-0.5 * float(diff @ diff) / (lengthscale * lengthscale)))


def markov_time_space_kernel(
    x: np.ndarray,
    t: int,
    y: np.ndarray,
    s: int,
    config: TVGPUCBConfig,
    spatial_kernel: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> float:
    """Product covariance k((x,t),(y,s)) from Bogunovic et al."""

    if t < 1 or s < 1:
        raise ValueError("paper time indices are 1-based and must be positive")
    kx = spatial_kernel(x, y) if spatial_kernel is not None else squared_exponential_kernel(x, y, config.lengthscale, config.signal_var)
    temporal = (1.0 - config.epsilon) ** (abs(int(t) - int(s)) / 2.0)
    return float(temporal * kx)


def finite_action_beta(t: int, num_actions: int, delta: float) -> float:
    """Finite-action confidence parameter beta_t = 2 log(|D| pi_t / delta).

    The standard GP-UCB finite-domain choice pi_t = pi^2 t^2 / 6 is used, so
    sum_t 1 / pi_t = 1. This is the finite-action confidence schedule used by
    the paper's GP-UCB-style acquisition.
    """

    if t < 1:
        raise ValueError("t must be positive")
    if num_actions < 1:
        raise ValueError("num_actions must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    pi_t = (np.pi * np.pi * t * t) / 6.0
    return float(2.0 * np.log(num_actions * pi_t / delta))


class TimeVaryingGPPosterior:
    """Exact GP posterior equations under the Markov time-varying kernel."""

    def __init__(
        self,
        config: TVGPUCBConfig,
        spatial_kernel: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        self.config = config
        self.spatial_kernel = spatial_kernel
        self.features: list[np.ndarray] = []
        self.times: list[int] = []
        self.observations: list[float] = []

    def update(self, x: np.ndarray, t: int, y: float) -> None:
        """Append observation y_t = f_t(x_t) + z_t."""

        if t < 1:
            raise ValueError("paper time indices are 1-based and must be positive")
        self.features.append(np.asarray(x, dtype=float).ravel().copy())
        self.times.append(int(t))
        self.observations.append(float(y))

    def predict(self, x: np.ndarray, t: int) -> tuple[float, float]:
        """Return posterior mean and latent variance at (x, t).

        Implements mu = k^T (K + sigma^2 I)^(-1) y and
        var = k((x,t),(x,t)) - k^T (K + sigma^2 I)^(-1) k.
        """

        if t < 1:
            raise ValueError("paper time indices are 1-based and must be positive")
        x = np.asarray(x, dtype=float).ravel()
        prior_var = markov_time_space_kernel(x, t, x, t, self.config, self.spatial_kernel)
        n = len(self.observations)
        if n == 0:
            return 0.0, prior_var

        y = np.asarray(self.observations, dtype=float)
        K = self.kernel_matrix()
        Ky = K + (self.config.noise_var + self.config.jitter) * np.eye(n)
        k = np.asarray(
            [markov_time_space_kernel(xi, ti, x, t, self.config, self.spatial_kernel) for xi, ti in zip(self.features, self.times)],
            dtype=float,
        )
        try:
            chol = np.linalg.cholesky(Ky)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
            v = np.linalg.solve(chol, k)
            mean = float(k @ alpha)
            var = float(prior_var - v @ v)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(Ky)
            mean = float(k @ inv @ y)
            var = float(prior_var - k @ inv @ k)
        return mean, max(var, 0.0)

    def kernel_matrix(self) -> np.ndarray:
        """Return K_{ij}=k((x_i,t_i),(x_j,t_j)) without observation noise."""

        n = len(self.observations)
        K = np.empty((n, n), dtype=float)
        for i, (xi, ti) in enumerate(zip(self.features, self.times)):
            for j, (xj, tj) in enumerate(zip(self.features, self.times)):
                K[i, j] = markov_time_space_kernel(xi, ti, xj, tj, self.config, self.spatial_kernel)
        return K

    def copy(self) -> "TimeVaryingGPPosterior":
        other = TimeVaryingGPPosterior(self.config, self.spatial_kernel)
        other.features = [x.copy() for x in self.features]
        other.times = list(self.times)
        other.observations = list(self.observations)
        return other


@dataclass(frozen=True)
class AcquisitionValue(Generic[ArmT]):
    arm: ArmT
    feature: np.ndarray
    mean: float
    variance: float
    beta: float
    score: float


class TVGPUCB(Generic[ArmT]):
    """Finite-action TV-GP-UCB reward maximization from Bogunovic et al."""

    name = "tv_gp_ucb"

    def __init__(
        self,
        actions: Sequence[ArmT],
        feature_map: Callable[[ArmT], np.ndarray],
        config: TVGPUCBConfig,
        spatial_kernel: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        if len(actions) == 0:
            raise ValueError("actions must be nonempty")
        self.actions = list(actions)
        self.feature_map = feature_map
        self.config = config
        self.posterior = TimeVaryingGPPosterior(config, spatial_kernel)
        self.t = 1
        self._last: AcquisitionValue[ArmT] | None = None

    def beta(self, t: int | None = None) -> float:
        return finite_action_beta(self.t if t is None else t, len(self.actions), self.config.delta)

    def acquisition_values(self, t: int | None = None) -> list[AcquisitionValue[ArmT]]:
        tt = self.t if t is None else int(t)
        beta = self.beta(tt)
        out: list[AcquisitionValue[ArmT]] = []
        for arm in self.actions:
            feature = np.asarray(self.feature_map(arm), dtype=float).ravel()
            mean, var = self.posterior.predict(feature, tt)
            score = mean + np.sqrt(beta) * np.sqrt(max(var, 0.0))
            out.append(AcquisitionValue(arm=arm, feature=feature, mean=mean, variance=var, beta=beta, score=float(score)))
        return out

    def select(self) -> ArmT:
        """Select argmax_x mu_{t-1}(x) + sqrt(beta_t) sigma_{t-1}(x)."""

        values = self.acquisition_values(self.t)
        self._last = max(values, key=lambda item: item.score)
        return self._last.arm

    def update(self, arm: ArmT, reward: float, t: int | None = None) -> None:
        """Update with realized reward for the selected finite-domain arm."""

        tt = self.t if t is None else int(t)
        feature = np.asarray(self.feature_map(arm), dtype=float).ravel()
        self.posterior.update(feature, tt, float(reward))
        self.t = max(self.t, tt + 1)


class TVGPLCB(TVGPUCB[ArmT]):
    """Cost-minimization counterpart using the TV-GP lower confidence bound."""

    name = "tv_gp_lcb"

    def acquisition_values(self, t: int | None = None) -> list[AcquisitionValue[ArmT]]:
        tt = self.t if t is None else int(t)
        beta = self.beta(tt)
        out: list[AcquisitionValue[ArmT]] = []
        for arm in self.actions:
            feature = np.asarray(self.feature_map(arm), dtype=float).ravel()
            mean, var = self.posterior.predict(feature, tt)
            score = mean - np.sqrt(beta) * np.sqrt(max(var, 0.0))
            out.append(AcquisitionValue(arm=arm, feature=feature, mean=mean, variance=var, beta=beta, score=float(score)))
        return out

    def select(self) -> ArmT:
        """Select argmin_x mu_{t-1}(x) - sqrt(beta_t) sigma_{t-1}(x)."""

        values = self.acquisition_values(self.t)
        self._last = min(values, key=lambda item: item.score)
        return self._last.arm

    def update(self, arm: ArmT, cost: float, t: int | None = None) -> None:
        """Update with realized cost for the selected finite-domain arm."""

        super().update(arm, cost, t)


class RGPUCB(TVGPUCB[ArmT]):
    """Auxiliary reset GP-UCB comparison baseline from the paper.

    This is not a requested primary baseline. It is provided only for comparisons
    that explicitly require R-GP-UCB. Within each batch it is stationary GP-UCB
    (epsilon=0); at batch boundaries the posterior is discarded.
    """

    name = "r_gp_ucb_auxiliary"

    def __init__(
        self,
        actions: Sequence[ArmT],
        feature_map: Callable[[ArmT], np.ndarray],
        config: TVGPUCBConfig,
        batch_size: int,
        spatial_kernel: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        stationary = TVGPUCBConfig(
            epsilon=0.0,
            noise_var=config.noise_var,
            delta=config.delta,
            lengthscale=config.lengthscale,
            signal_var=config.signal_var,
            jitter=config.jitter,
        )
        super().__init__(actions, feature_map, stationary, spatial_kernel)
        self.batch_size = int(batch_size)

    def update(self, arm: ArmT, reward: float, t: int | None = None) -> None:
        super().update(arm, reward, t)
        if (self.t - 1) % self.batch_size == 0:
            self.posterior = TimeVaryingGPPosterior(self.config, self.posterior.spatial_kernel)


@dataclass(frozen=True)
class RealizedCostArm(Generic[ArmT, ContextT]):
    """A feasible root action together with the context used to featurize it."""

    action: ArmT
    context: ContextT


class RealizedCostBanditAdapter(Generic[ArmT, ContextT]):
    """Adapter from finite feasible control actions/contexts to TV-GP-LCB arms.

    The adapter intentionally treats each feasible root action in the current
    context as a bandit arm and updates the GP only with the observed realized
    cost supplied by the caller after the environment step. It does not add
    rollout costs, model-predicted terminal costs, nominal simulator costs, or
    other control heuristics; such terms would be outside the Bogunovic et al.
    finite-action bandit algorithm and must be layered externally if desired.
    """

    def __init__(
        self,
        action_provider: Callable[[ContextT], Sequence[ArmT]],
        feature_map: Callable[[ContextT, ArmT], np.ndarray],
        config: TVGPUCBConfig,
        spatial_kernel: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        self.action_provider = action_provider
        self.feature_map = feature_map
        self.config = config
        self.spatial_kernel = spatial_kernel
        self.bandit: TVGPLCB[ArmT] | None = None
        self.posterior = TimeVaryingGPPosterior(self.config, self.spatial_kernel)
        self.t = 1
        self._last_arm: RealizedCostArm[ArmT, ContextT] | None = None

    def select(self, context: ContextT) -> ArmT:
        actions = list(self.action_provider(context))
        if len(actions) == 0:
            raise ValueError("action_provider returned no feasible actions")
        action = self._select_with_history(context, actions)
        self._last_arm = RealizedCostArm(action=action, context=context)
        return action

    def _select_with_history(self, context: ContextT, actions: Sequence[ArmT]) -> ArmT:
        # Rebuild a temporary finite-domain selector for the current feasible set,
        # preserving all past observations in the time-varying posterior.
        current = TVGPLCB(
            actions=actions,
            feature_map=lambda action: self.feature_map(context, action),
            config=self.config,
            spatial_kernel=self.spatial_kernel,
        )
        current.posterior = self.posterior.copy()
        current.t = self.t
        action = current.select()
        self.bandit = current
        return action

    def update(self, realized_cost: float, action: ArmT | None = None, context: ContextT | None = None) -> None:
        """Record the realized cost of the chosen action/context arm."""

        if action is None or context is None:
            if self._last_arm is None:
                raise ValueError("no previous selection is available; pass action and context explicitly")
            action = self._last_arm.action
            context = self._last_arm.context
        feature = np.asarray(self.feature_map(context, action), dtype=float).ravel()
        self.posterior.update(feature, self.t, float(realized_cost))
        self.t += 1
