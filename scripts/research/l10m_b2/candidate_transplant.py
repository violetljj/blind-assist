"""Freeze and execute the deterministic B2 seed-89 candidate transplant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.research.l10m_b1.evaluator import evaluate_spec
from scripts.research.l10m_b1.policy_space import (
    INITIAL_SPEC,
    canonical_spec,
    changed_components,
    parse_raw,
    parse_structured,
    render_raw,
    render_structured,
)


PROTOCOL_ID = "L10M-B2-SEED89-CANDIDATE-TRANSPLANT-V1"
SOURCE_RUN_ID = "b1-20260820T115002-98733875"
SOURCE_SEED = 89
SOURCE_ARM = "raw"
SOURCE_GENERATION = 2
SOURCE_CANDIDATE_SHA256 = "3dceae4c4003fb9bbf864f414ef316cf06d649a57280da1c4f0bdd2fa93cb919"
SOURCE_EVIDENCE_SHA256 = {
    "events.jsonl": "b6f6e4ff3bccab2e46d0252e017fd7be73d4be533fd1233f0a011e94583e9ed6",
    "execution_manifest.json": "de3b02f0515baa18a9ef226c15bc20f0b98f28e6820dd504b0519d1c3b1b228b",
    "result_adjudication_r1.json": "35b0f816c281c4a3680eacf0aaaf15305864143db790f5aa433405f66ef7f72a",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_source(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    run_dir = run_dir.resolve()
    for name, expected in SOURCE_EVIDENCE_SHA256.items():
        actual = _sha256(run_dir / name)
        if actual != expected:
            raise RuntimeError(f"source evidence hash mismatch for {name}: {actual}")

    adjudication = json.loads((run_dir / "result_adjudication_r1.json").read_text(encoding="utf-8"))
    if (
        adjudication.get("run_id") != SOURCE_RUN_ID
        or adjudication.get("terminal") != "B1_EVALUABLE_COMPLETE"
        or adjudication.get("scientific_verdict") != "B1_INCONCLUSIVE"
    ):
        raise RuntimeError("B2 source is not the frozen evaluable B1 closure")

    matches = []
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if (
            event.get("kind") == "completion"
            and event.get("seed") == SOURCE_SEED
            and event.get("arm") == SOURCE_ARM
            and event.get("generation") == SOURCE_GENERATION
        ):
            matches.append(event)
    if len(matches) != 1 or matches[0].get("candidate_output_sha256") != SOURCE_CANDIDATE_SHA256:
        raise RuntimeError("frozen seed-89 Raw candidate is missing or ambiguous")
    return adjudication, matches[0]


def build_protocol(run_dir: Path) -> dict[str, object]:
    adjudication, source = _load_source(run_dir)
    source_spec = parse_raw(str(source["candidate_output"]))
    if changed_components(INITIAL_SPEC, source_spec) != ["action_selection"]:
        raise RuntimeError("source candidate is not the localized action_selection intervention")
    if float(source["behavioral_score"]) != 0.993103448275862:
        raise RuntimeError("source candidate score differs from the B1 receipt")

    return {
        "protocol_id": PROTOCOL_ID,
        "status": "B2_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED",
        "research_question": "Does the seed-89 Raw action_selection intervention retain identical semantics and behavior when encoded through the Structured interface with search disabled?",
        "claim_ceiling": "causal localization inside the frozen finite B1 policy interface and hidden synthetic evaluator only; not evidence about general structured representations, model search quality, end-to-end behavior, devices, users, or safety effects",
        "source_b1": {
            "run_id": SOURCE_RUN_ID,
            "terminal": adjudication["terminal"],
            "scientific_verdict": adjudication["scientific_verdict"],
            "seed": SOURCE_SEED,
            "arm": SOURCE_ARM,
            "generation": SOURCE_GENERATION,
            "candidate_output_sha256": SOURCE_CANDIDATE_SHA256,
            "evidence_sha256": SOURCE_EVIDENCE_SHA256,
        },
        "frozen_intervention": {
            "canonical_spec": canonical_spec(source_spec),
            "canonical_spec_sha256": _sha256_bytes(canonical_spec(source_spec).encode("utf-8")),
            "changed_components": ["action_selection"],
            "action_selection_turn_threshold": source_spec.action_selection_turn_threshold,
        },
        "execution": {
            "search_enabled": False,
            "model_calls": 0,
            "candidate_generation": "deterministic render_raw/render_structured from the same frozen PolicySpec",
            "evaluation": "parse each representation, then evaluate with the same frozen B1 evaluator and hidden cohort",
        },
        "verdict_rules": {
            "B2_SEARCH_PATH_FAILURE_SIGNAL": "both encodings round-trip to the frozen canonical intervention and produce identical full evaluator output and behavior hash",
            "B2_REPRESENTATION_BOTTLENECK_SIGNAL": "both encodings are admitted as the same intervention but Structured produces different frozen-evaluator behavior or score",
            "B2_NOT_EVALUABLE_TRANSLATION": "the Structured encoding cannot be admitted or does not preserve the frozen canonical intervention",
        },
    }


def execute(protocol_path: Path, run_dir: Path) -> dict[str, object]:
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_protocol = build_protocol(run_dir)
    if protocol != expected_protocol:
        raise RuntimeError("protocol does not match the frozen B1 source evidence")

    _, source = _load_source(run_dir)
    source_spec = parse_raw(str(source["candidate_output"]))
    raw_encoding = render_raw(source_spec)
    structured_encoding = render_structured(source_spec)

    raw_spec = parse_raw(raw_encoding)
    try:
        structured_spec = parse_structured(structured_encoding)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return {
            "protocol_id": PROTOCOL_ID,
            "terminal": "B2_EVALUABLE_COMPLETE",
            "scientific_verdict": "B2_NOT_EVALUABLE_TRANSLATION",
            "translation_error": str(error),
            "protocol_sha256": _sha256(protocol_path),
        }

    source_canonical = canonical_spec(source_spec)
    raw_canonical = canonical_spec(raw_spec)
    structured_canonical = canonical_spec(structured_spec)
    semantic_equivalent = source_canonical == raw_canonical == structured_canonical
    if not semantic_equivalent:
        verdict = "B2_NOT_EVALUABLE_TRANSLATION"
        raw_evaluation = None
        structured_evaluation = None
    else:
        raw_evaluation = evaluate_spec(raw_spec)
        structured_evaluation = evaluate_spec(structured_spec)
        verdict = (
            "B2_SEARCH_PATH_FAILURE_SIGNAL"
            if raw_evaluation == structured_evaluation
            else "B2_REPRESENTATION_BOTTLENECK_SIGNAL"
        )

    behavior_hashes = {
        "raw": None if raw_evaluation is None else _sha256_bytes(_canonical_json(raw_evaluation).encode("utf-8")),
        "structured": None if structured_evaluation is None else _sha256_bytes(_canonical_json(structured_evaluation).encode("utf-8")),
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "terminal": "B2_EVALUABLE_COMPLETE",
        "scientific_verdict": verdict,
        "claim_ceiling": protocol["claim_ceiling"],
        "source_b1": protocol["source_b1"],
        "search_calls": 0,
        "model_calls": 0,
        "semantic_equivalent": semantic_equivalent,
        "canonical_spec_sha256": {
            "source": _sha256_bytes(source_canonical.encode("utf-8")),
            "raw": _sha256_bytes(raw_canonical.encode("utf-8")),
            "structured": _sha256_bytes(structured_canonical.encode("utf-8")),
        },
        "encoding_sha256": {
            "raw": _sha256_bytes(raw_encoding.encode("utf-8")),
            "structured": _sha256_bytes(structured_encoding.encode("utf-8")),
        },
        "behavior_sha256": behavior_hashes,
        "behavior_equal": raw_evaluation == structured_evaluation if semantic_equivalent else False,
        "raw_evaluation": raw_evaluation,
        "structured_evaluation": structured_evaluation,
        "protocol_sha256": _sha256(protocol_path),
    }


def _write_create_once(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--b1-run-dir", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--b1-run-dir", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        payload = build_protocol(args.b1_run_dir)
    else:
        payload = execute(args.protocol, args.b1_run_dir)
    _write_create_once(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
