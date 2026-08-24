"""Synthetic OCR-stage development harness for SAGE-R V2.

This module tests the algorithm after OCR.  It compares the V1-style
substring plus two-frame debounce baseline with a relational semantic-anchor
graph, an explicit NONE hypothesis, quality-aware UNKNOWN handling, and
correlated-evidence suppression.  The generated observations are synthetic;
they are not camera or OCR measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "SEMANTIC_ANCHOR_GRAPH_AND_BELIEF_V2"
SCHEMA_VERSION = "blindassist_semantic_anchor_graph_and_belief_v2"
CLAIM_CEILING = (
    "SYNTHETIC_OCR_STAGE_DEVELOPMENT_RELATIONAL_IDENTITY_AND_OPEN_SET_BELIEF_"
    "NO_NATURAL_OCR_CAMERA_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(character for character in value if character.isalnum() or character == "?")


def _edit_similarity(left: str, right: str) -> float:
    left, right = normalize_text(left), normalize_text(right)
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            substitution = 0 if left_character == right_character else 0.35 if "?" in {left_character, right_character} else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + substitution,
                )
            )
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(len(left), len(right)))


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def horizontal_overlap(self, other: "Box") -> float:
        overlap = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        return overlap / max(1e-6, min(self.x1 - self.x0, other.x1 - other.x0))


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    box: Box


@dataclass(frozen=True)
class Token:
    text: str
    box: Box
    confidence: float = 0.95


@dataclass(frozen=True)
class Frame:
    episode_id: str
    frame_index: int
    viewpoint: str
    candidates: tuple[Candidate, ...]
    tokens: tuple[Token, ...]
    blur: float
    perspective: float
    truth: str
    expected_state: str
    note: str


@dataclass(frozen=True)
class TargetGraph:
    tokens: tuple[str, ...]
    relation: str = "SAME_LINE"
    anchor_relation: str = "ABOVE_CANDIDATE"


def _softmax(logits: Mapping[str, float]) -> dict[str, float]:
    maximum = max(logits.values())
    values = {key: math.exp(value - maximum) for key, value in logits.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def _token_quality(token: Token, frame: Frame) -> float:
    size = min(1.0, token.box.height / 0.035)
    return token.confidence * size * (1.0 - frame.blur) * (1.0 - 0.65 * frame.perspective)


def frame_observability(frame: Frame) -> float:
    if not frame.tokens:
        return max(0.0, (1.0 - frame.blur) * (1.0 - frame.perspective) * 0.45)
    return max(_token_quality(token, frame) for token in frame.tokens)


def _line_groups(tokens: Sequence[Token]) -> list[list[Token]]:
    rows: list[list[Token]] = []
    for token in sorted(tokens, key=lambda item: (item.box.cy, item.box.x0)):
        group = next(
            (
                candidate
                for candidate in rows
                if abs(sum(item.box.cy for item in candidate) / len(candidate) - token.box.cy)
                <= max(0.018, token.box.height * 0.8)
            ),
            None,
        )
        if group is None:
            rows.append([token])
        else:
            group.append(token)
    groups: list[list[Token]] = []
    for row in rows:
        current: list[Token] = []
        for token in sorted(row, key=lambda item: item.box.x0):
            if current and token.box.x0 - current[-1].box.x1 > 0.035:
                groups.append(current)
                current = []
            current.append(token)
        if current:
            groups.append(current)
    return groups


def _group_box(group: Sequence[Token]) -> Box:
    return Box(
        min(token.box.x0 for token in group),
        min(token.box.y0 for token in group),
        max(token.box.x1 for token in group),
        max(token.box.y1 for token in group),
    )


def _association(group: Sequence[Token], candidate: Candidate) -> float:
    box = _group_box(group)
    overlap = box.horizontal_overlap(candidate.box)
    vertical_gap = candidate.box.y0 - box.y1
    above = math.exp(-abs(vertical_gap - 0.025) / 0.10) if vertical_gap >= -0.02 else 0.05
    center = math.exp(-abs(box.cx - candidate.box.cx) / 0.16)
    return 0.50 * overlap + 0.30 * above + 0.20 * center


def _distinctiveness(expected: str, groups: Sequence[Sequence[Token]]) -> float:
    if not groups:
        return 0.0
    appearances = sum(
        1
        for group in groups
        if any(_edit_similarity(expected, token.text) >= 0.72 for token in group)
    )
    lexical_specificity = min(1.0, max(0.25, len(normalize_text(expected)) / 4))
    return lexical_specificity * (math.log((len(groups) + 1) / (appearances + 0.5)) / math.log(len(groups) + 1))


def graph_candidate_scores(target: TargetGraph, frame: Frame) -> dict[str, dict[str, float]]:
    """Score candidate identity from lexical, layout, association and quality evidence."""
    groups = _line_groups(frame.tokens)
    token_weights = {
        expected: max(0.05, _distinctiveness(expected, groups))
        for expected in target.tokens
    }
    scores: dict[str, dict[str, float]] = {}
    for candidate in frame.candidates:
        best = {
            "score": 0.0,
            "lexical": 0.0,
            "layout": 0.0,
            "association": 0.0,
            "quality": 0.0,
            "distinctiveness": 0.0,
        }
        for group in groups:
            matches = []
            matched_tokens = []
            for expected in target.tokens:
                token = max(group, key=lambda item: _edit_similarity(expected, item.text))
                matches.append(_edit_similarity(expected, token.text))
                matched_tokens.append(token)
            weight_sum = sum(token_weights.values())
            lexical = sum(
                token_weights[expected] * match
                for expected, match in zip(target.tokens, matches, strict=True)
            ) / max(1e-6, weight_sum)
            if len(matched_tokens) > 1:
                line_delta = max(token.box.cy for token in matched_tokens) - min(token.box.cy for token in matched_tokens)
                order = all(
                    matched_tokens[index].box.cx <= matched_tokens[index + 1].box.cx
                    for index in range(len(matched_tokens) - 1)
                )
                layout = math.exp(-line_delta / 0.025) * (1.0 if order else 0.55)
            else:
                layout = 1.0
            association = _association(group, candidate)
            quality = sum(_token_quality(token, frame) for token in matched_tokens) / len(matched_tokens)
            distinctive = sum(token_weights.values()) / len(token_weights)
            decisive_index = max(range(len(target.tokens)), key=lambda index: token_weights[target.tokens[index]])
            decisive_match = matches[decisive_index]
            exact_suffix_penalty = 1.0
            normalized_line = "".join(normalize_text(token.text) for token in group)
            normalized_target = "".join(normalize_text(token) for token in target.tokens)
            if normalized_target in normalized_line and normalized_line != normalized_target:
                exact_suffix_penalty = 0.62
            score = (
                0.40 * lexical
                + 0.20 * layout
                + 0.28 * association
                + 0.12 * quality
            ) * (0.65 + 0.35 * distinctive) * exact_suffix_penalty
            if decisive_match < 0.78:
                score *= 0.52
            if score > best["score"]:
                best = {
                    "score": score,
                    "lexical": lexical,
                    "layout": layout,
                    "association": association,
                    "quality": quality,
                    "distinctiveness": distinctive,
                    "decisive_match": decisive_match,
                }
        scores[candidate.candidate_id] = best
    return scores


def _burst_signature(frame: Frame) -> str:
    token_state = sorted(
        (normalize_text(token.text), round(token.box.cx, 2), round(token.box.cy, 2), round(token.confidence, 1))
        for token in frame.tokens
    )
    payload = json.dumps([frame.viewpoint, token_state, round(frame.blur, 1), round(frame.perspective, 1)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ReferentBelief:
    def __init__(self, candidate_ids: Iterable[str]):
        self.candidate_ids = tuple(sorted(candidate_ids))
        uniform = 1 / (len(self.candidate_ids) + 1)
        self.probabilities = {candidate_id: uniform for candidate_id in self.candidate_ids}
        self.probabilities["NONE"] = uniform
        self.last_signature: str | None = None
        self.reset_after_unknown = False

    def update(self, frame: Frame, scores: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
        observability = frame_observability(frame)
        signature = _burst_signature(frame)
        novel = signature != self.last_signature
        self.last_signature = signature
        if observability < 0.28:
            self.reset_after_unknown = True
            return self._decision("UNKNOWN", scores, observability, novel, 0.0)

        if self.reset_after_unknown:
            uniform = 1 / len(self.probabilities)
            self.probabilities = {key: uniform for key in self.probabilities}
            self.reset_after_unknown = False

        maximum = max((item["score"] for item in scores.values()), default=0.0)
        evidence_strength = min(1.0, max(0.0, (observability - 0.25) / 0.65))
        novelty_weight = 1.0 if novel else 0.08
        update_weight = evidence_strength * novelty_weight
        likelihood_logits = {
            candidate_id: 8.0 * (scores[candidate_id]["score"] - 0.48)
            for candidate_id in self.candidate_ids
        }
        likelihood_logits["NONE"] = 7.0 * (0.58 - maximum) + 1.0 * observability
        likelihood = _softmax(likelihood_logits)
        posterior = {
            key: max(1e-9, self.probabilities[key]) * max(1e-9, likelihood[key]) ** update_weight
            for key in self.probabilities
        }
        total = sum(posterior.values())
        self.probabilities = {key: value / total for key, value in posterior.items()}

        ranked = sorted(
            ((key, value) for key, value in self.probabilities.items() if key != "NONE"),
            key=lambda item: (-item[1], item[0]),
        )
        best_id, best_probability = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        best_frame_score = scores[best_id]["score"]
        if (
            best_probability >= 0.52
            and best_probability - max(runner_up, self.probabilities["NONE"]) >= 0.18
            and best_frame_score >= 0.64
        ):
            state = "TARGET"
        elif self.probabilities["NONE"] >= 0.58 and observability >= 0.58:
            state = "NONE"
        else:
            state = "UNCERTAIN"
        return self._decision(state, scores, observability, novel, update_weight)

    def _decision(
        self,
        state: str,
        scores: Mapping[str, Mapping[str, float]],
        observability: float,
        novel: bool,
        update_weight: float,
    ) -> dict[str, Any]:
        candidates = sorted(
            ((key, value) for key, value in self.probabilities.items() if key != "NONE"),
            key=lambda item: (-item[1], item[0]),
        )
        selected = candidates[0][0] if state == "TARGET" else None
        return {
            "state": state,
            "selected_candidate": selected,
            "probabilities": dict(sorted(self.probabilities.items())),
            "scores": scores,
            "observability": observability,
            "source_novel": novel,
            "update_weight": update_weight,
        }


class SubstringFsmBaseline:
    """V1-like string match assigned to the nearest candidate, then two-frame debounce."""

    def __init__(self, target: TargetGraph):
        self.needle = "".join(normalize_text(token) for token in target.tokens)
        self.pending: str | None = None
        self.count = 0

    def update(self, frame: Frame) -> dict[str, Any]:
        matches = []
        for group in _line_groups(frame.tokens):
            line = "".join(normalize_text(token.text) for token in group)
            if self.needle and self.needle in line:
                box = _group_box(group)
                candidate = min(
                    frame.candidates,
                    key=lambda item: (abs(item.box.cx - box.cx) + abs(item.box.y0 - box.cy), item.candidate_id),
                )
                matches.append(candidate.candidate_id)
        matched = matches[0] if len(set(matches)) == 1 else None
        if matched is None:
            self.pending, self.count = None, 0
        elif matched == self.pending:
            self.count += 1
        else:
            self.pending, self.count = matched, 1
        locked = self.pending if self.count >= 2 else None
        return {
            "state": "TARGET" if locked else "ABSTAIN",
            "selected_candidate": locked,
            "matches": matches,
            "consecutive_matches": self.count,
        }


def _door(candidate_id: str, x0: float) -> Candidate:
    return Candidate(candidate_id, Box(x0, 0.32, x0 + 0.22, 0.95))


def _sign(texts: Sequence[str], x: float, y: float, *, height: float = 0.045, confidence: float = 0.95) -> tuple[Token, ...]:
    widths = [max(0.055, 0.018 * len(text)) for text in texts]
    tokens = []
    cursor = x - (sum(widths) + 0.012 * (len(widths) - 1)) / 2
    for text, width in zip(texts, widths, strict=True):
        tokens.append(Token(text, Box(cursor, y, cursor + width, y + height), confidence))
        cursor += width + 0.012
    return tuple(tokens)


def _jitter(tokens: Sequence[Token], rng: random.Random, amount: float = 0.004) -> tuple[Token, ...]:
    output = []
    for token in tokens:
        dx, dy = rng.uniform(-amount, amount), rng.uniform(-amount, amount)
        output.append(
            Token(
                token.text,
                Box(token.box.x0 + dx, token.box.y0 + dy, token.box.x1 + dx, token.box.y1 + dy),
                token.confidence,
            )
        )
    return tuple(output)


def generate_cohort(seed: int = 302) -> list[Frame]:
    """Generate fixed mechanism-targeted episodes with deterministic geometric jitter."""
    rng = random.Random(seed)
    a, b, c = _door("A", 0.08), _door("B", 0.39), _door("C", 0.70)
    candidates = (a, b, c)
    frames: list[Frame] = []

    def add(
        episode: str,
        viewpoint: str,
        tokens: Sequence[Token],
        truth: str,
        expected: str,
        note: str,
        *,
        blur: float = 0.05,
        perspective: float = 0.08,
        repeat_signature: bool = False,
    ) -> None:
        index = sum(frame.episode_id == episode for frame in frames)
        jittered = tuple(tokens) if repeat_signature else _jitter(tokens, rng)
        frames.append(Frame(episode, index, viewpoint, candidates, jittered, blur, perspective, truth, expected, note))

    standard = _sign(("ROOM", "301"), a.box.cx, 0.245) + _sign(("ROOM", "302"), b.box.cx, 0.245) + _sign(("ROOM", "320"), c.box.cx, 0.245)
    add("adjacent_rooms", "front-1", standard, "B", "TARGET", "three adjacent same-prefix rooms")
    add("adjacent_rooms", "front-2", standard, "B", "TARGET", "fresh stable view")

    directory = _sign(("ROOM", "302"), a.box.cx, 0.39) + _sign(("ROOM", "30?"), b.box.cx, 0.245) + _sign(("ROOM", "303"), c.box.cx, 0.245)
    add("directory_binding", "oblique-1", directory, "B", "UNCERTAIN", "target text on directory is not above a door", perspective=0.28)
    add("directory_binding", "oblique-2", directory, "B", "TARGET", "partial physical sign plus relational support", perspective=0.20)
    clear_target = _sign(("ROOM", "302"), b.box.cx, 0.245) + _sign(("ROOM", "303"), c.box.cx, 0.245)
    add("directory_binding", "close-3", clear_target, "B", "TARGET", "closer view resolves target sign")

    suffix = _sign(("ROOM", "302A"), a.box.cx, 0.245) + _sign(("ROOM", "302"), b.box.cx, 0.245) + _sign(("ROOM", "320"), c.box.cx, 0.245)
    add("suffix_hard_negative", "front-1", suffix, "B", "TARGET", "302A must not inherit 302 identity")
    add("suffix_hard_negative", "front-2", suffix, "B", "TARGET", "fresh suffix view")

    absent = _sign(("ROOM", "301"), a.box.cx, 0.245) + _sign(("ROOM", "303"), b.box.cx, 0.245) + _sign(("ROOM", "320"), c.box.cx, 0.245)
    add("absent_clear", "front-1", absent, "NONE", "NONE", "clear high-quality absence observation")
    add("absent_clear", "front-2", absent, "NONE", "NONE", "independent clear absence observation")

    tiny = _sign(("ROOM", "30?"), b.box.cx, 0.29, height=0.009, confidence=0.42)
    add("unreadable_not_negative", "far-static", tiny, "B", "UNKNOWN", "unreadable OCR must not count as absence", blur=0.72, perspective=0.55, repeat_signature=True)
    add("unreadable_not_negative", "far-static", tiny, "B", "UNKNOWN", "correlated unreadable repeat", blur=0.72, perspective=0.55, repeat_signature=True)

    spurious = _sign(("ROOM", "302"), a.box.cx, 0.40, height=0.032, confidence=0.72)
    for _ in range(6):
        add("correlated_directory_burst", "static-directory", spurious, "NONE", "UNCERTAIN", "same directory read repeated", blur=0.20, perspective=0.22, repeat_signature=True)
    add("correlated_directory_burst", "wide-clear", absent, "NONE", "NONE", "new source geometry establishes absence")

    add("reacquisition", "front-start", standard, "B", "TARGET", "initial target evidence")
    add("reacquisition", "front-confirm", standard, "B", "TARGET", "initial independent confirmation")
    add("reacquisition", "occluded", (), "B", "UNKNOWN", "occlusion is unknown", blur=0.86, perspective=0.35)
    reacquired = _sign(("ROOM", "301"), a.box.cx, 0.245) + _sign(("ROOM", "302"), c.box.cx, 0.245)
    add("reacquisition", "right-sweep-1", reacquired, "C", "TARGET", "candidate identity moves after reacquisition")
    add("reacquisition", "right-sweep-2", reacquired, "C", "TARGET", "fresh view re-establishes identity")
    return frames


def _row_correct(row: Mapping[str, Any], arm: str) -> bool:
    decision = row[arm]
    truth = row["truth"]
    if truth == "NONE":
        return decision["state"] == "NONE"
    return decision["state"] == "TARGET" and decision["selected_candidate"] == truth


def _metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    target_rows = [row for row in rows if row["truth"] != "NONE"]
    absent_rows = [row for row in rows if row["truth"] == "NONE"]
    wrong_locks = sum(
        row[arm]["state"] == "TARGET" and row[arm]["selected_candidate"] != row["truth"]
        for row in rows
    )
    return {
        "frames": len(rows),
        "correct_terminal_frames": sum(_row_correct(row, arm) for row in rows),
        "target_correct_locks": sum(_row_correct(row, arm) for row in target_rows),
        "target_frames": len(target_rows),
        "wrong_locks": wrong_locks,
        "none_correct": sum(_row_correct(row, arm) for row in absent_rows),
        "absent_frames": len(absent_rows),
        "unknown_preserved": sum(
            row["expected_state"] == "UNKNOWN" and row[arm]["state"] == "UNKNOWN"
            for row in rows
        ),
        "unknown_frames": sum(row["expected_state"] == "UNKNOWN" for row in rows),
    }


def evaluate(seed: int = 302) -> tuple[dict[str, Any], dict[str, Any]]:
    target = TargetGraph(("ROOM", "302"))
    frames = generate_cohort(seed)
    rows = []
    for episode_id in sorted({frame.episode_id for frame in frames}):
        episode = [frame for frame in frames if frame.episode_id == episode_id]
        baseline = SubstringFsmBaseline(target)
        belief = ReferentBelief(candidate.candidate_id for candidate in episode[0].candidates)
        for frame in episode:
            scores = graph_candidate_scores(target, frame)
            rows.append(
                {
                    "episode_id": frame.episode_id,
                    "frame_index": frame.frame_index,
                    "viewpoint": frame.viewpoint,
                    "truth": frame.truth,
                    "expected_state": frame.expected_state,
                    "note": frame.note,
                    "frame": asdict(frame),
                    "baseline": baseline.update(frame),
                    "v2": belief.update(frame, scores),
                }
            )
    rows.sort(key=lambda row: (row["episode_id"], row["frame_index"]))
    baseline_metrics = _metrics(rows, "baseline")
    v2_metrics = _metrics(rows, "v2")
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "SYNTHETIC_OCR_STAGE_DEVELOPMENT",
        "seed": seed,
        "question": (
            "Does relational anchor-to-candidate scoring plus open-set, quality-aware, burst-suppressed belief "
            "outperform substring plus two-frame debounce on controlled semantic distractors?"
        ),
        "metrics": {"substring_fsm": baseline_metrics, "sage_r_v2": v2_metrics},
        "delta": {
            "correct_terminal_frames": v2_metrics["correct_terminal_frames"] - baseline_metrics["correct_terminal_frames"],
            "wrong_locks": v2_metrics["wrong_locks"] - baseline_metrics["wrong_locks"],
            "none_correct": v2_metrics["none_correct"] - baseline_metrics["none_correct"],
            "unknown_preserved": v2_metrics["unknown_preserved"] - baseline_metrics["unknown_preserved"],
        },
        "mechanisms": [
            "relational lexical-layout-association-quality graph score",
            "scene-adaptive semantic distinctiveness",
            "explicit candidate and NONE posterior",
            "low-observability UNKNOWN without negative update",
            "source-signature correlated evidence suppression",
        ],
        "interpretation_boundary": [
            "All observations are deterministic synthetic OCR-stage tokens and geometry; this is not an OCR or camera benchmark.",
            "The cohort is mechanism-targeted Development evidence and was designed with knowledge of the proposed failure modes.",
            "Thresholds are Development constants and are not open-set calibration evidence on a natural distribution.",
            "No Android, active-perception policy, navigation, safety, or product behavior is evaluated.",
        ],
        "claim_ceiling": CLAIM_CEILING,
    }
    return {"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "rows": rows}, report


def _render_html(raw: Mapping[str, Any], report: Mapping[str, Any], path: Path) -> None:
    metric_rows = []
    for key in ("correct_terminal_frames", "target_correct_locks", "wrong_locks", "none_correct", "unknown_preserved"):
        metric_rows.append(
            f"<tr><td>{key}</td><td>{report['metrics']['substring_fsm'][key]}</td>"
            f"<td>{report['metrics']['sage_r_v2'][key]}</td></tr>"
        )
    episode_rows = []
    for row in raw["rows"]:
        truth = row["truth"]
        baseline = row["baseline"]["state"] + (f":{row['baseline']['selected_candidate']}" if row["baseline"]["selected_candidate"] else "")
        v2 = row["v2"]["state"] + (f":{row['v2']['selected_candidate']}" if row["v2"]["selected_candidate"] else "")
        v2_class = "good" if _row_correct(row, "v2") or v2 == row["expected_state"] else "neutral"
        episode_rows.append(
            f"<tr><td>{row['episode_id']}</td><td>{row['frame_index']}</td><td>{truth}</td>"
            f"<td>{baseline}</td><td class='{v2_class}'>{v2}</td><td>{row['note']}</td></tr>"
        )
    html = f"""<!doctype html><meta charset='utf-8'><title>SAGE-R V2 controlled result</title>
<style>body{{font:15px system-ui;margin:32px;background:#10141c;color:#eaf0f8}}h1{{color:#7ee7c4}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #344154;padding:8px;text-align:left}}th{{background:#202b3b}}.good{{color:#7ee7c4;font-weight:700}}.neutral{{color:#f2c96d}}code{{color:#9cc5ff}}</style>
<h1>SAGE-R V2: graph + open-set belief</h1>
<p>Controlled synthetic OCR-stage development result. Claim ceiling: <code>{CLAIM_CEILING}</code></p>
<h2>Headline metrics</h2><table><tr><th>metric</th><th>substring + FSM</th><th>SAGE-R V2</th></tr>{''.join(metric_rows)}</table>
<h2>Sequence decisions</h2><table><tr><th>episode</th><th>frame</th><th>truth</th><th>baseline</th><th>V2</th><th>mechanism</th></tr>{''.join(episode_rows)}</table>
"""
    path.write_text(html, encoding="utf-8", newline="\n")


def run(run_dir: Path, seed: int = 302) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite run: {run_dir}")
    run_dir.mkdir(parents=True)
    raw, report = evaluate(seed)
    _atomic_json(run_dir / "raw-decisions.json", raw)
    report["raw_decisions_sha256"] = _sha256_file(run_dir / "raw-decisions.json")
    _atomic_json(run_dir / "final-report.json", report)
    _render_html(raw, report, run_dir / "result.html")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=302)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.run_dir.resolve(), args.seed)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
