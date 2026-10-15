"""Validate and summarize generated observations without reading field truth."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(root):
    path = root / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len({r["image"] for r in rows}) == len(rows), "Duplicate image result"
    good = [r for r in rows if r.get("error") is None]
    for r in good:
        assert set(r["fields"]) == {"product", "brand", "variety"}
        assert all(isinstance(field["text"], str) for field in r["fields"].values())
        assert "".join(token["text"] for token in r["tokens"]) == r["raw_text"]
        for field in r["fields"].values():
            values = [r["tokens"][i]["logprob"] for i in field["token_indices"]]
            if values:
                assert abs(statistics.mean(values) - field["mean_logprob"]) < 1e-7
                assert min(values) == field["min_logprob"]
    terminals = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("terminal_*.json")]
    times = sorted(r["inference_s"] for r in good)
    return {"images": len(rows), "successes": len(good), "errors": len(rows) - len(good),
            "nonempty_fields": {field: sum(bool(r["fields"][field]["text"].strip()) for r in good) for field in ["product", "brand", "variety"]},
            "inference_total_s": sum(times), "inference_median_s": statistics.median(times),
            "inference_p95_s_nearest_rank": times[max(0, math.ceil(len(times) * .95) - 1)],
            "process_total_wall_s": sum(t["process_wall_s"] for t in terminals),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in good),
            "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config_file_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
            "canonical_config_sha256": rows[0]["config_sha256"] if rows else None,
            "terminal_files": [p.name for p in root.glob("terminal_*.json")],
            "validation": "unique images, three string fields, token-text reconstruction, field logprob recomputation; no labels read"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--arms", nargs="+", default=["image_only", "ocr_conditioned"])
    parser.add_argument("--summary-name", default="summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    result = {}
    for name in args.arms:
        if (root / name / "predictions.jsonl").exists():
            result[name] = summarize(root / name)
    result["total_gpu_process_wall_s"] = sum(r["process_total_wall_s"] for r in result.values())
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    (root / args.summary_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
