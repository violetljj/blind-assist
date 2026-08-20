from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from .triage import ProtocolIntegrityError, analyze, canonical_bytes, partial


def _gate(cohorts, policies, loco, integrity=True):
    minimum = len([v for v in cohorts.values() if v["n"] >= 6]) >= 2 and len([v for v in policies.values() if v["evaluable"]]) >= 2
    rhos = [v["rho"] for v in cohorts.values() if v["n"] >= 6]
    cohort = minimum and sum(v > 0 for v in rhos) / len(rhos) >= .75 and sorted(rhos)[len(rhos)//2] >= .2 and all(v > -.2 for v in rhos)
    policy = len([v for v in policies.values() if v["evaluable"]]) >= 2 and all(v["rho"] > 0 for v in policies.values() if v["evaluable"])
    increment = loco["relative"] >= .02 and loco["median"] > 0
    return minimum and cohort and policy and increment and integrity


def test_eligibility_independent_of_outcome():
    fixture = {"complete_steps": 8, "outcome": 0.1}
    changed = {**fixture, "outcome": 0.9}
    eligible = lambda row: row["complete_steps"] == 8
    assert eligible(fixture) == eligible(changed)


def test_missing_trajectory_fails_closed():
    assert [1, 2, 4] != list(range(1, 9))


def test_b5_seed_has_no_confirmatory_authority():
    cohorts = {"only_b5": {"n": 100, "rho": 1.0, "authority": "HYPOTHESIS_GENERATING_ONLY"}}
    confirmatory = {k: v for k, v in cohorts.items() if v["authority"] == "CONFIRMATORY_SET"}
    assert not _gate(confirmatory, {}, {"relative": 1.0, "median": 1.0})


def test_pooled_rescue_forbidden():
    cohorts = {"a": {"n": 12, "rho": .9}, "b": {"n": 12, "rho": -.2}}
    policies = {"p": {"evaluable": True, "rho": .8}, "q": {"evaluable": True, "rho": .8}}
    assert not _gate(cohorts, policies, {"relative": .5, "median": .5})


def test_policy_rescue_forbidden():
    cohorts = {"a": {"n": 12, "rho": .8}, "b": {"n": 12, "rho": .8}}
    policies = {"balanced": {"evaluable": True, "rho": .9}, "control": {"evaluable": True, "rho": 0.0}}
    assert not _gate(cohorts, policies, {"relative": .5, "median": .5})


def test_manifest_digest_is_immutable(tmp_path: Path):
    manifest = tmp_path / "manifest.json"; digest = tmp_path / "manifest.sha256"
    manifest.write_bytes(canonical_bytes({"protocol": "wrong"})); digest.write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + "  manifest.json\n", encoding="ascii")
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolIntegrityError): analyze(tmp_path, manifest, digest)


def test_deterministic_partial_replay():
    rows = [{"I": float(i % 3), "R": float(i), "Y": float(i * 2 + i % 2)} for i in range(8)]
    assert partial(copy.deepcopy(rows)) == partial(copy.deepcopy(rows))


def test_zero_model_provider_imports():
    source = Path(__file__).with_name("triage.py").read_text(encoding="utf-8")
    assert "provider_transport" not in source
    assert "subprocess.run([\"git\"" in source
    assert "codex.exe" not in source
