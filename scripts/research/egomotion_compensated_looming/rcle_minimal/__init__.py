"""RCLE-Minimal Phase A synthetic-only research implementation."""

from .protocol import PROTOCOL_SHA256, TrialSpec, enumerate_trials, load_protocol

__all__ = [
    "PROTOCOL_SHA256",
    "TrialSpec",
    "enumerate_trials",
    "load_protocol",
]
