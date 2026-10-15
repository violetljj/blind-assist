"""LFVQA annotation replay: retain answer/failure/advice under two sentences.

This is an annotation-role diagnostic, NOT factuality or BLV user evaluation.
Only sentence text enters the learned selector. Image-group split keeps every
model's answer to the same image together. No third-party package is required.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import time
import urllib.request

REVISION = "358d12904635bbf612558da10febfe5e35967540"
URL = f"https://raw.githubusercontent.com/UT-CS-HCI/lfvqa/{REVISION}/data/all.json"
EXPECTED_SHA = "823f0282868b9766cb47d22de37cf65d638d352ac7eb2768c708732a55096cb9"
ROLES = ("Answer", "Answer Failure", "Suggestion")


def words(sentence):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())


def features(sentence):
    tokens = words(sentence)
    return collections.Counter(tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])])


class RoleNB:
    def __init__(self, rows):
        self.models = []
        self.vocab = set()
        for row in rows:
            for sentence in row["sentences"]:
                self.vocab.update(features(sentence))
        for role in ROLES:
            token_counts = [collections.Counter(), collections.Counter()]
            doc_counts = [0, 0]
            for row in rows:
                for sentence, roles in zip(row["sentences"], row["roles"]):
                    target = int(role in roles)
                    token_counts[target].update(features(sentence))
                    doc_counts[target] += 1
            totals = [sum(c.values()) + len(self.vocab) for c in token_counts]
            self.models.append((token_counts, doc_counts, totals))

    def predict(self, sentence):
        f = {k: v for k, v in features(sentence).items() if k in self.vocab}
        result = []
        for counts, docs, totals in self.models:
            logp = [math.log((docs[c] + 1) / (sum(docs) + 2)) for c in (0, 1)]
            for c in (0, 1):
                logp[c] += sum(n * math.log((counts[c][w] + 1) / totals[c]) for w, n in f.items())
            logodds = max(-40, min(40, logp[1] - logp[0]))
            result.append(1 / (1 + math.exp(-logodds)))
        return result

    def select(self, sentences, budget=2):
        probs = [self.predict(s) for s in sentences]
        # Greedy expected coverage of the three roles; no evaluation labels.
        selected = []
        uncovered = [1.0] * len(ROLES)
        for _ in range(min(budget, len(sentences))):
            candidates = [i for i in range(len(sentences)) if i not in selected]
            best = max(candidates, key=lambda i: (sum(a * b for a, b in zip(uncovered, probs[i])), -i))
            selected.append(best)
            uncovered = [a * (1 - b) for a, b in zip(uncovered, probs[best])]
        return sorted(selected)


def evaluate(rows, selections):
    counters = collections.Counter()
    for row, selected in zip(rows, selections):
        present = set(itertools.chain.from_iterable(row["roles"])) & set(ROLES)
        retained = set(itertools.chain.from_iterable(row["roles"][i] for i in selected))
        for role in present:
            counters[role + "_eligible"] += 1
            counters[role + "_retained"] += int(role in retained)
        counters["eligible_role_sets"] += int(bool(present))
        counters["complete_role_sets"] += int(bool(present) and present <= retained)
        counters["selected_words"] += sum(len(words(row["sentences"][i])) for i in selected)
        counters["full_words"] += sum(len(words(s)) for s in row["sentences"])
        counters["selected_sentences"] += len(selected)
        counters["full_sentences"] += len(row["sentences"])
    return {"answers": len(rows), **dict(counters)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts.local/work/assistive-thesis-screen-20261015/answer_compression"))
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "all.json"
    downloaded = 0
    if not source.exists():
        with urllib.request.urlopen(URL, timeout=30) as response:
            payload = response.read(10_000_001)
        if len(payload) > 10_000_000:
            raise RuntimeError("Source exceeds pilot bound")
        source.write_bytes(payload)
        downloaded = len(payload)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        raise RuntimeError("Source bytes differ from recorded public release")
    data = json.loads(source.read_text(encoding="utf-8"))
    rows, invalid = [], []
    for question_id, item in data.items():
        # Stable image-level holdout, including all generators and question IDs.
        image_id = item["image_url"]
        split = "eval" if int(hashlib.sha256(image_id.encode()).hexdigest(), 16) % 5 == 0 else "train"
        for model, answer in item["long_answers"].items():
            sentences, labels = answer["answer_sentences"], answer["functional_roles"]
            if len(sentences) != len(labels) or not sentences:
                invalid.append({"question_id": question_id, "model": model, "sentences": len(sentences), "labels": len(labels)})
                continue
            rows.append({"question_id": question_id, "image_id": image_id, "model": model, "split": split,
                         "sentences": sentences, "roles": labels})
    train = [r for r in rows if r["split"] == "train"]
    evaluation = [r for r in rows if r["split"] == "eval"]
    assert {r["image_id"] for r in train}.isdisjoint({r["image_id"] for r in evaluation})
    model = RoleNB(train)
    baseline = [list(range(min(2, len(r["sentences"])))) for r in evaluation]
    selected = [model.select(r["sentences"]) for r in evaluation]
    opportunity = [i for i, r in enumerate(evaluation) if len(r["sentences"]) > 2]
    subset = [evaluation[i] for i in opportunity]
    summary = {
        "evidence": "Consumed public LFVQA role-annotation replay; no new VLM generation or image reasoning",
        "source": {"url": URL, "revision": REVISION, "sha256": digest, "bytes": source.stat().st_size},
        "questions": len(data), "unique_images": len({v["image_url"] for v in data.values()}),
        "answer_records": sum(len(v["long_answers"]) for v in data.values()),
        "length_mismatch_or_empty": len(invalid), "aligned_records": len(rows),
        "train_answers": len(train), "eval_answers": len(evaluation),
        "train_images": len({r["image_id"] for r in train}), "eval_images": len({r["image_id"] for r in evaluation}),
        "models": dict(collections.Counter(r["model"] for r in rows)),
        "first_two": evaluate(evaluation, baseline), "learned_two": evaluate(evaluation, selected),
        "opportunity_first_two": evaluate(subset, [baseline[i] for i in opportunity]),
        "opportunity_learned_two": evaluate(subset, [selected[i] for i in opportunity]),
        "per_generator": {m: {
            "first_two": evaluate([r for r in evaluation if r["model"] == m], [s for r, s in zip(evaluation, baseline) if r["model"] == m]),
            "learned_two": evaluate([r for r in evaluation if r["model"] == m], [s for r, s in zip(evaluation, selected) if r["model"] == m]),
        } for m in sorted({r["model"] for r in evaluation})},
        "backend": "Python stdlib CPU; TASK_NOT_GPU_SUITABLE sparse text counts",
        "new_download_bytes_this_execution": downloaded,
        "wall_seconds": time.perf_counter() - start,
        "limitations": [
            "Equal sentence count is not equal words or speaking duration; word counts are reported.",
            "Functional role is not factual correctness, useful advice, or user preference.",
            "Length mismatches are excluded, never silently zipped or repaired.",
            "Even length-aligned annotations can be semantically noisy; no manual relabeling performed.",
            "Held-out images share generator families; no unseen-model or Chinese-language claim.",
            "Naive Bayes is a feasibility baseline, not a modern summarization strong baseline.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "invalid_records.json").write_text(json.dumps(invalid, indent=2), encoding="utf-8")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, b, s in zip(evaluation, baseline, selected):
            f.write(json.dumps({"question_id": row["question_id"], "image_id": row["image_id"], "model": row["model"],
                                "first_two": b, "learned_two": s, "sentence_count": len(row["sentences"])}) + "\n")
    (args.output / "split_manifest.json").write_text(json.dumps({r["question_id"]: {"image_id": r["image_id"], "split": r["split"]} for r in rows}, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("questions", "answer_records", "length_mismatch_or_empty", "train_answers", "eval_answers", "opportunity_first_two", "opportunity_learned_two", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
