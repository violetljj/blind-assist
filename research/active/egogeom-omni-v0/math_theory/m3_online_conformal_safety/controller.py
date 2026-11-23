"""Causal delayed-feedback quantile controller. One update token per issued frame."""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path

import numpy as np
from numba import njit

OUT = Path(__file__).resolve().parents[5] / "artifacts.local/work/egogeom-omni-v0/math_theory/m3_online_conformal_safety"
HORIZON_M = 12.0


@njit
def initialize(horizon, slots, epsilon):
    per_lane = (horizon + slots - 1) // slots
    levels = int(np.floor(np.log2(per_lane))) + 1
    prior = np.empty(levels + 2)
    rate = np.empty(levels + 2)
    step = np.empty(levels)
    for j in range(levels):
        length = 2 ** j
        prior[j] = 0.5 / levels / ((per_lane + length - 1) // length)
        rate[j] = min(0.5, np.sqrt(-np.log(prior[j]) / length)) * epsilon / 4
        step[j] = 4.0 * epsilon / np.sqrt(length)
    prior[levels:] = 0.25
    # The two virtual advisers move by at most one, even though the action
    # domain has diameter four. Their worst-case excess is only 1/epsilon.
    rate[levels:] = min(0.5, np.sqrt(np.log(4.0) / per_lane)) * epsilon
    mass = np.ones((slots, levels + 2)) * prior
    advice = np.zeros_like(mass)
    return prior, rate, step, mass, advice


@njit
def issue(t, prior, rate, mass, advice):
    slots, size = mass.shape
    lane, index = t % slots, t // slots
    levels = size - 2
    weighted_sum, base = 0.0, 0.0
    for j in range(levels):
        if index % (2 ** j) == 0:
            advice[lane, j] = 0.0
            mass[lane, j] = prior[j]
        weight = rate[j] * mass[lane, j]
        weighted_sum += weight * advice[lane, j]
        base += weight
    plus = mass[lane, levels] * rate[levels]
    minus = mass[lane, levels + 1] * rate[levels + 1]
    q = (weighted_sum + plus - minus) / base
    if q > 1.0:
        q = (weighted_sum + 2 * plus - minus) / (base + plus)
    elif q < -1.0:
        q = (weighted_sum + plus - 2 * minus) / (base + minus)
    advice[lane, levels] = min(2.0, q + 1.0)
    advice[lane, levels + 1] = max(-2.0, q - 1.0)
    return q


@njit
def learn(t, q, gradient, mass, advice, rate, step):
    lane = t % mass.shape[0]
    for j in range(len(rate)):
        excess = gradient * (advice[lane, j] - q)
        mass[lane, j] *= 1.0 + rate[j] * excess
        if j < len(step):
            advice[lane, j] = min(2.0, max(-2.0, advice[lane, j] + step[j] * gradient))


@njit
def dr_gradient(q, alpha, propensity, prob_error, observed, signed_score):
    if q >= 1.0:
        return -alpha
    if q <= -1.0:
        return 1.0 - alpha
    g = prob_error - alpha
    if observed:
        g += (int(signed_score > q) - prob_error) / propensity
    return g


def standardized(prediction_m, sigma_m, truth_m):
    residual = (np.clip(prediction_m, 0, HORIZON_M) - np.clip(truth_m, 0, HORIZON_M)) / sigma_m
    return residual / (1.0 + abs(residual))


def lower_distance(prediction_m, sigma_m, q):
    if q >= 1.0:
        return 0.0
    if q <= -1.0:
        return float("inf")
    return max(0.0, min(prediction_m, HORIZON_M) - sigma_m * q / (1.0 - abs(q)))


@dataclass(frozen=True)
class Prediction:
    frame: int
    distance_m: float
    sigma_m: float
    q: float
    lower_m: float
    propensity: float
    prob_error: float

    @property
    def action_lower_m(self):
        """An empty set on the capped range supplies no usable lower bound."""
        return self.lower_m if self.lower_m <= HORIZON_M else 0.0


class RiskConformalController:
    """Call predict, then reveal that token at/after issue and before its lane is reused.

    Native validity is disclosed only at feedback. If the randomized audit did
    not reveal the outcome, pass truth_m=None. Calibration targets capped range.
    """
    def __init__(self, horizon, alpha=0.05, max_delay=5, epsilon=0.7):
        if horizon < 1 or max_delay < 0 or not (0 < epsilon <= 1 and 0 < alpha < 1):
            raise ValueError("invalid horizon, delay, propensity floor or target")
        self.horizon = horizon
        self.alpha = alpha
        self.epsilon = epsilon
        self.slots = max_delay + 1
        self.prior, self.rate, self.step_size, self.mass, self.advice = initialize(horizon, self.slots, epsilon)
        self.next_frame = 0
        self.pending = [None] * self.slots
        self.resolved = 0
        self.proxy_sum = 0.0

    def predict(self, distance_m, sigma_m, *, propensity=1.0, prob_error=None):
        t = self.next_frame
        slot = t % self.slots
        if t >= self.horizon or self.pending[slot] is not None:
            raise RuntimeError("horizon exceeded or older feedback has not arrived by its deadline")
        p = self.alpha if prob_error is None else float(prob_error)
        if not (self.epsilon <= propensity <= 1 and 0 <= p <= 1 and
                np.isfinite(sigma_m) and sigma_m > 0 and distance_m >= 0):
            raise ValueError("invalid prediction or audit design")
        q = issue(t, self.prior, self.rate, self.mass, self.advice)
        distance_m = min(float(distance_m), HORIZON_M)
        token = Prediction(t, distance_m, sigma_m, q, lower_distance(distance_m, sigma_m, q), propensity, p)
        self.pending[slot] = token
        self.next_frame += 1
        return token

    def reveal(self, token, valid, truth_m=None):
        slot = token.frame % self.slots
        if self.pending[slot] is not token:
            raise ValueError("unknown, duplicate, or stale feedback")
        if valid:
            if truth_m is not None and not truth_m >= 0:
                raise ValueError("invalid disclosed range")
            score = 0.0 if truth_m is None else standardized(token.distance_m, token.sigma_m, truth_m)
            g = dr_gradient(token.q, self.alpha, token.propensity, token.prob_error, truth_m is not None, score)
            learn(token.frame, token.q, g, self.mass, self.advice, self.rate, self.step_size)
            self.proxy_sum += self.alpha + g
            self.resolved += 1
        self.pending[slot] = None

    def audit_upper(self, delta=0.05):
        """Known MAR propensity; fixed valid count under exogenous audit, finite-prefix risk.

        Bounds realized average miscoverage on valid resolved tokens. The
        all-prefix union allocation allows reading this value at any prefix.
        """
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0,1)")
        n = self.resolved
        if not n:
            return 1.0
        if self.epsilon == 1.0:
            return min(1.0, max(0.0, self.proxy_sum / n))
        # Audit times must be outcome-independent or conditioned on a fixed reveal schedule.
        x = log(n * (n + 1) / delta)
        variance = n * (1.0 / self.epsilon - 1.0)
        b = 1.0 / self.epsilon
        return min(1.0, max(0.0, (self.proxy_sum + sqrt(2 * variance * x) + 2 * b * x / 3) / n))
