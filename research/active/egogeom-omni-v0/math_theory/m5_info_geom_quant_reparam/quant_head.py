"""Terminal, positive normal-range disparity codecs for the frozen v4 heads."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ASSETS = ROOT / "artifacts.local/work/egogeom-omni-v0"
OUT = ASSETS / "math_theory" / Path(__file__).resolve().parent.name
LO, HI = 0.15, 25.0
CENTER = np.sqrt(LO * HI)
HALF_SPAN = np.log(HI / LO) / 2


def gauss_pairs():
    nodes, weights = np.polynomial.legendre.leggauss(16)
    return [(float(t * t), float(2 * w)) for t, w in zip(nodes, weights) if t > 0]


def logcode(disparity):
    """Approximate log(Z/CENTER), using only rational arithmetic."""
    cd = CENTER * disparity
    t = (1 - cd) / (1 + cd)
    t2 = t * t
    return t * sum(w / (1 - q * t2) for q, w in gauss_pairs())


def decode(log_centered):
    """[2/2] Pade with range reduction and five squarings."""
    x = log_centered / 32
    even = 12 + x * x
    odd = 6 * x
    z = (even + odd) / (even - odd)
    for _ in range(5):
        z = z * z
    return CENTER * z


def mixed_primitive(z):
    # w=1/Z^2 + 1/Z^4, with lengths numerically expressed in metres.
    return np.arcsinh(z) - np.hypot(z, 1.0) / z


def int_log_roundtrip(d):
    code = np.clip(np.rint((logcode(d) + HALF_SPAN) * (255 / (2 * HALF_SPAN))), 0, 255)
    return decode(code * (2 * HALF_SPAN / 255) - HALF_SPAN)
