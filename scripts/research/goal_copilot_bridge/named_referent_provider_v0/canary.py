"""Public, consumed-by-this-canary storefront mechanics acquisition and run."""

from __future__ import annotations

import html
import json
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .provider import (
    DINOv2ReferenceAdapter,
    GroundingDINOProposalAdapter,
    MapBearingAdapter,
    NamedReferentProviderV0,
    PaddleOCRV5Adapter,
)
from .schema import CLAIM_CEILING, CurrentFrame, GoalReferencePack, sha256_file


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Mozilla/5.0 BlindAssist-NamedReferentProviderV0/0.1 (engineering canary; no redistribution bundle)"
CORPUS = (
    {
        "case_id": "starbucks_dazaifu_reference",
        "title": "File:Starbucks Coffee Dazaifutenmangu Omotesando Store-1.jpg",
        "role": "TARGET_REFERENCE",
        "goal_group": "starbucks_dazaifu",
    },
    {
        "case_id": "starbucks_dazaifu_query_a",
        "title": "File:Starbucks Coffee Dazaifutenmangu Omotesando Store-2.jpg",
        "role": "SAME_POI_QUERY_DIFFERENT_VIEW",
        "goal_group": "starbucks_dazaifu",
    },
    {
        "case_id": "starbucks_dazaifu_query_b",
        "title": "File:Starbucks Coffee Dazaifutenmangu Omotesando Store-3.jpg",
        "role": "SAME_POI_QUERY_DIFFERENT_VIEW",
        "goal_group": "starbucks_dazaifu",
    },
    {
        "case_id": "starbucks_hudson_distractor",
        "title": "File:14th St 9th Av Hudson St td (2022-02-07) 25 - Starbucks (678 Hudson Street).jpg",
        "role": "SAME_BRAND_DIFFERENT_POI_DISTRACTOR",
        "goal_group": "starbucks_hudson",
    },
    {
        "case_id": "tsuiwah_tko_reference",
        "title": "File:HK TKO 將軍澳 Tseung Kwan O 尚德廣場 Sheung Tak Estate Shopping Centre shop Tsui Wah Restaurant March 2022 Px3 01.jpg",
        "role": "TARGET_REFERENCE",
        "goal_group": "tsuiwah_tko",
    },
    {
        "case_id": "tsuiwah_tko_query",
        "title": "File:HK TKO Spot mall 將軍澳 Tseung Kwan O 尚德廣場 Sheung Tak Estate Shopping Centre shop 翠華餐廳 Tsui Wah Restaurant September 2022 Px3 05.jpg",
        "role": "SAME_POI_QUERY_DIFFERENT_DATE_VIEW",
        "goal_group": "tsuiwah_tko",
    },
    {
        "case_id": "tsuiwah_tsuenwan_distractor",
        "title": "File:HK TWD 荃灣 Tsuen Wan 圓敦圍 Yuen Tun Circuit 荃灣廣場 Tsuen Wan Plaza mall shop Tsui Wah Ping Teng October 2021 SS2 01.jpg",
        "role": "SAME_BRAND_DIFFERENT_POI_DISTRACTOR",
        "goal_group": "tsuiwah_tsuenwan",
    },
    {
        "case_id": "starbucks_logo",
        "title": "File:Starbucks logo.jpg",
        "role": "LOGO_REFERENCE",
        "goal_group": "starbucks_brand",
    },
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _plain(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(value))).strip()


def _request_json(parameters: Mapping[str, str]) -> Mapping[str, Any]:
    url = COMMONS_API + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code != 429:
            raise
        completed = subprocess.run(
            ["curl.exe", "-L", "--fail", "--silent", "--show-error", "-A", USER_AGENT, url],
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Commons API failed after curl fallback: {completed.stderr.strip()}") from error
        return json.loads(completed.stdout)


def _download(url: str, path: Path) -> None:
    # Commons appends analytics query parameters to imageinfo thumbnail URLs.
    # The byte resource is the same without them and the plain CDN URL avoids
    # spurious throttling observed on repeated exact-title acquisition.
    parts = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 6):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            os.replace(temporary, path)
            return
        except urllib.error.HTTPError as error:
            if error.code == 429:
                completed = subprocess.run(
                    [
                        "curl.exe", "-L", "--fail", "--silent", "--show-error",
                        "--retry", "2", "--retry-delay", "5", "-A", USER_AGENT,
                        "-o", str(temporary), url,
                    ],
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if completed.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                    os.replace(temporary, path)
                    return
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = min(20.0, float(retry_after)) if retry_after and retry_after.isdigit() else 3.0 * attempt
            time.sleep(delay)
    raise RuntimeError(f"download retries exhausted: {url}")


def acquire_public_canary(artifact_root: Path) -> Path:
    """Download exact Commons titles and preserve their live license metadata."""

    root = artifact_root.resolve()
    images = root / "data" / "images"
    images.mkdir(parents=True, exist_ok=True)
    payload = _request_json({
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": "|".join(spec["title"] for spec in CORPUS),
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": "1280",
    })
    pages = payload.get("query", {}).get("pages", {})
    pages_by_title = {page.get("title"): page for page in pages.values() if isinstance(page, Mapping)}
    records = []
    for spec in CORPUS:
        page = pages_by_title.get(spec["title"])
        if not isinstance(page, Mapping):
            raise RuntimeError(f"Commons title did not resolve uniquely: {spec['title']}")
        info = page.get("imageinfo", [None])[0]
        if not isinstance(info, Mapping):
            raise RuntimeError(f"Commons title has no imageinfo: {spec['title']}")
        metadata = info.get("extmetadata", {})
        download_url = str(info.get("thumburl") or info.get("url"))
        suffix = ".png" if spec["title"].lower().endswith(".png") else ".jpg"
        local_path = images / f"{spec['case_id']}{suffix}"
        if not local_path.is_file():
            _download(download_url, local_path)
        with Image.open(local_path) as opened:
            width, height = opened.size
            opened.verify()
        records.append({
            **spec,
            "local_path": str(local_path),
            "sha256": sha256_file(local_path),
            "width": width,
            "height": height,
            "source_page_url": str(info.get("descriptionurl")),
            "download_url": download_url,
            "original_url": str(info.get("url")),
            "license_short_name": _plain(metadata.get("LicenseShortName", {}).get("value")),
            "license_url": _plain(metadata.get("LicenseUrl", {}).get("value")),
            "artist": _plain(metadata.get("Artist", {}).get("value")),
            "credit": _plain(metadata.get("Credit", {}).get("value")),
            "usage_terms": _plain(metadata.get("UsageTerms", {}).get("value")),
            "attribution_required": _plain(metadata.get("AttributionRequired", {}).get("value")),
            "access_role": "PUBLIC_REPEATABLE_CANARY_CONTENT_INSPECTED_NOT_FRESH_NOT_CONFIRMATION",
        })
        time.sleep(0.5)
    manifest = {
        "schema_version": "blindassist_named_referent_public_canary_manifest_v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "Wikimedia Commons API exact-title acquisition",
        "data_role": "PUBLIC_REPEATABLE_CANARY_CONTENT_INSPECTED_NOT_FRESH_NOT_FORMAL",
        "claim_ceiling": CLAIM_CEILING,
        "redistribution": "NOT_BUNDLED_IN_GIT; follow each source license and trademark terms",
        "records": records,
    }
    manifest_path = root / "data" / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = {record["case_id"]: record for record in value["records"]}
    for record in records.values():
        local = Path(record["local_path"])
        if not local.is_file() or sha256_file(local) != record["sha256"]:
            raise RuntimeError(f"canary data missing or changed: {record['case_id']}")
    return value, records


def _scale_ladder(source: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
    for scale in (1.0, 0.75, 0.50, 0.35):
        width = max(1, round(rgb.width * scale))
        height = max(1, round(rgb.height * scale))
        resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", rgb.size, (127, 127, 127))
        offset = ((rgb.width - width) // 2, (rgb.height - height) // 2)
        canvas.paste(resized, offset)
        path = output_dir / f"tsuiwah-scale-{scale:.2f}.jpg"
        canvas.save(path, quality=92)
        records.append({
            "scale": scale,
            "path": str(path),
            "sha256": sha256_file(path),
            "canvas_size": [rgb.width, rgb.height],
            "content_size": [width, height],
            "role": "DERIVED_CHINESE_REMOTE_TEXT_SCALE_STRESS",
        })
    return records


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median_ms": None, "max_ms": None}
    return {"count": len(values), "median_ms": statistics.median(values), "max_ms": max(values)}


def _ocr_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(result["latency_ms"]) for result in results]
    class_counts = {name: 0 for name in ("EXACT", "SUBSTRING", "FUZZY", "NONE")}
    polygon_count = 0
    per_image = []
    for result in results:
        best_class = "NONE"
        best_ratio = 0.0
        texts = []
        for item in result.get("items", []):
            match = item["normalized_match"]["best"]
            class_counts[match["match_class"]] += 1
            if item["source"].get("polygon") is not None:
                polygon_count += 1
            texts.append(item["raw_match"]["recognized_text"])
            if match["fuzzy_ratio"] > best_ratio:
                best_ratio = match["fuzzy_ratio"]
                best_class = match["match_class"]
        per_image.append({
            "frame_id": result.get("_frame_id"),
            "status": result["status"],
            "latency_ms": result["latency_ms"],
            "item_count": len(result.get("items", [])),
            "best_match_class": best_class,
            "best_fuzzy_ratio": best_ratio,
            "recognized_texts": texts,
            "error": result.get("error"),
        })
    return {
        "statuses": {status: sum(result["status"] == status for result in results) for status in ("AVAILABLE", "NOT_EVALUABLE")},
        "latency": _latency_summary(latencies),
        "match_class_item_counts": class_counts,
        "polygon_item_count": polygon_count,
        "per_image": per_image,
    }


def run_vision_canary(*, artifact_root: Path, dino_model_dir: Path, reference_model_dir: Path) -> Path:
    """Run torch-backed providers in a process isolated from Paddle CUDA DLLs."""

    root = artifact_root.resolve()
    manifest_path = root / "data" / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = acquire_public_canary(root)
    _, records = _load_manifest(manifest_path)
    evidence_dir = root / "evidence"
    outputs_dir = evidence_dir / "provider_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    goal = GoalReferencePack.from_mapping({
        "name": "Starbucks Dazaifu",
        "aliases": ["Starbucks", "STARBUCKS COFFEE"],
        "reference_images": [records["starbucks_dazaifu_reference"]["local_path"]],
        "logo": records["starbucks_logo"]["local_path"],
        "map_bearing_degrees": 92.0,
    })
    frame = CurrentFrame.from_mapping({
        "frame_id": "starbucks-dazaifu-query-a",
        "image_path": records["starbucks_dazaifu_query_a"]["local_path"],
        "heading_degrees": 80.0,
    })
    reference = DINOv2ReferenceAdapter(model_dir=reference_model_dir, device="cuda")
    proposal = GroundingDINOProposalAdapter(model_dir=dino_model_dir)
    provider_output = NamedReferentProviderV0({
        "text_evidence": None,
        "visual_reference_evidence": reference,
        "proposal_evidence": proposal,
        "bearing_evidence": MapBearingAdapter(),
    }).run(frame, goal)
    _write_json(outputs_dir / "vision-provider-output.json", provider_output)
    ranking = []
    for case_id in (
        "starbucks_dazaifu_query_a",
        "starbucks_dazaifu_query_b",
        "starbucks_hudson_distractor",
        "tsuiwah_tko_query",
        "tsuiwah_tsuenwan_distractor",
    ):
        candidate_frame = CurrentFrame.from_mapping({"frame_id": case_id, "image_path": records[case_id]["local_path"]})
        channel = reference.collect(candidate_frame, GoalReferencePack.from_mapping({
            "name": "Starbucks Dazaifu",
            "aliases": ["Starbucks"],
            "reference_images": [records["starbucks_dazaifu_reference"]["local_path"]],
            "logo": None,
        }))
        similarity = channel["items"][0]["raw_match"]["cosine_similarity"] if channel["status"] == "AVAILABLE" and channel["items"] else None
        ranking.append({
            "case_id": case_id,
            "role": records[case_id]["role"],
            "goal_group": records[case_id]["goal_group"],
            "status": channel["status"],
            "similarity": similarity,
            "latency_ms": channel["latency_ms"],
            "error": channel["error"],
        })
    ranking.sort(key=lambda item: (item["similarity"] is None, -(item["similarity"] or -1.0), item["case_id"]))
    for index, item in enumerate(ranking, start=1):
        item["rank"] = index
    result = {
        "schema_version": "blindassist_named_referent_vision_canary_v0",
        "claim_ceiling": CLAIM_CEILING,
        "execution_isolation": "TORCH_PROCESS_SEPARATE_FROM_PADDLE_CUDA_DLLS",
        "grounding_dino_model_dir": str(dino_model_dir.resolve()),
        "reference_model_dir": str(reference_model_dir.resolve()),
        "provider_output_path": str(outputs_dir / "vision-provider-output.json"),
        "provider_output": provider_output,
        "reference_identity": reference.identity(),
        "ranking": ranking,
    }
    path = evidence_dir / "vision-canary.json"
    _write_json(path, result)
    reference.close()
    return path


def run_canary(
    *,
    artifact_root: Path,
    dino_model_dir: Path,
    reference_model_dir: Path,
    ocr_devices: Sequence[str] = ("cpu", "gpu:0"),
) -> tuple[Path, Path]:
    root = artifact_root.resolve()
    manifest_path = root / "data" / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = acquire_public_canary(root)
    manifest, records = _load_manifest(manifest_path)
    evidence_dir = root / "evidence"
    outputs_dir = evidence_dir / "provider_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "models" / "paddlex"

    starbucks_goal = GoalReferencePack.from_mapping({
        "name": "Starbucks Dazaifu",
        "aliases": ["Starbucks", "STARBUCKS COFFEE"],
        "reference_images": [records["starbucks_dazaifu_reference"]["local_path"]],
        "logo": records["starbucks_logo"]["local_path"],
        "map_bearing_degrees": 92.0,
    })
    query_frame = CurrentFrame.from_mapping({
        "frame_id": "starbucks-dazaifu-query-a",
        "image_path": records["starbucks_dazaifu_query_a"]["local_path"],
        "heading_degrees": 80.0,
    })

    vision_path = evidence_dir / "vision-canary.json"
    if not vision_path.is_file():
        raise RuntimeError(
            "vision-canary.json is required; run the run-vision-canary command with the project Python first "
            "to isolate torch from Paddle CUDA DLLs"
        )
    vision = json.loads(vision_path.read_text(encoding="utf-8"))
    if Path(vision.get("grounding_dino_model_dir", "")).resolve() != dino_model_dir.resolve():
        raise RuntimeError("vision canary Grounding DINO model path does not match this run")
    if Path(vision.get("reference_model_dir", "")).resolve() != reference_model_dir.resolve():
        raise RuntimeError("vision canary reference model path does not match this run")
    vision_output = vision["provider_output"]
    primary_ocr = PaddleOCRV5Adapter(device=ocr_devices[-1], cache_dir=cache_dir)
    provider = NamedReferentProviderV0({
        "text_evidence": primary_ocr,
        "visual_reference_evidence": None,
        "proposal_evidence": None,
        "bearing_evidence": MapBearingAdapter(),
    })
    text_output = provider.run(query_frame, starbucks_goal)
    _write_json(outputs_dir / "text-provider-output.json", text_output)
    combined = dict(text_output)
    combined["execution_shards"] = [
        {"backend": "paddle", "path": str(outputs_dir / "text-provider-output.json")},
        {"backend": "torch", "path": vision["provider_output_path"]},
    ]
    combined["evidence"] = dict(text_output["evidence"])
    combined["evidence"]["visual_reference_evidence"] = vision_output["evidence"]["visual_reference_evidence"]
    combined["evidence"]["proposal_evidence"] = vision_output["evidence"]["proposal_evidence"]
    _write_json(outputs_dir / "combined-independent-provider-output.json", combined)
    ranking = vision["ranking"]

    ladder = _scale_ladder(Path(records["tsuiwah_tko_query"]["local_path"]), root / "data" / "derived")
    tsui_goal = GoalReferencePack.from_mapping({
        "name": "翠華餐廳",
        "aliases": ["翠華", "Tsui Wah", "Tsui Wah Restaurant"],
        "reference_images": [records["tsuiwah_tko_reference"]["local_path"]],
        "logo": None,
    })
    ocr_by_device: dict[str, Any] = {}
    for device in ocr_devices:
        adapter = primary_ocr if device == ocr_devices[-1] else PaddleOCRV5Adapter(device=device, cache_dir=cache_dir)
        results = []
        work = [
            ("starbucks-base", Path(records["starbucks_dazaifu_query_a"]["local_path"]), starbucks_goal),
            ("starbucks-logo", Path(records["starbucks_logo"]["local_path"]), starbucks_goal),
            *( (f"tsuiwah-scale-{entry['scale']:.2f}", Path(entry["path"]), tsui_goal) for entry in ladder ),
        ]
        for frame_id, path, goal in work:
            result = dict(adapter.collect(CurrentFrame.from_mapping({"frame_id": frame_id, "image_path": path}), goal))
            result["_frame_id"] = frame_id
            results.append(result)
            _write_json(outputs_dir / f"ocr-{re.sub(r'[^a-zA-Z0-9]+', '-', device)}-{frame_id}.json", result)
        ocr_by_device[device] = _ocr_summary(results)
        if adapter is not primary_ocr:
            adapter.close()

    provider_status = {
        channel: {
            "status": combined["evidence"][channel]["status"],
            "latency_ms": combined["evidence"][channel]["latency_ms"],
            "item_count": len(combined["evidence"][channel]["items"]),
            "error": combined["evidence"][channel]["error"],
            "provider_identity": combined["evidence"][channel]["provider_identity"],
        }
        for channel in combined["evidence"]
    }
    target_ranks = [item["rank"] for item in ranking if item["goal_group"] == "starbucks_dazaifu" and item["similarity"] is not None]
    result = {
        "schema_version": "blindassist_named_referent_provider_canary_result_v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "REVERSIBLE_EXPLORATION",
        "profile": "CANARY_LITE",
        "benchmark_role": "PLATFORM_ENGINEERING_CANARY",
        "claim_ceiling": CLAIM_CEILING,
        "scientific_performance_claim": "FORBIDDEN_NOT_MADE",
        "default_app_or_model_promotion": "FORBIDDEN_NOT_PERFORMED",
        "fresh_or_formal_cohort": False,
        "question": "Can independent named-POI evidence adapters execute and preserve inspectable provenance/error semantics?",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "provider_status": provider_status,
        "ocr_canary": {
            "devices": ocr_by_device,
            "scale_ladder": ladder,
            "match_semantics_exercised": ["EXACT", "SUBSTRING", "FUZZY"],
            "note": "match classes are diagnostics over real OCR output, not truth or identification authority",
        },
        "reference_image_canary": {
            "model": vision["reference_identity"],
            "ranking": ranking,
            "same_poi_candidate_ranks": target_ranks,
            "best_same_poi_rank": min(target_ranks) if target_ranks else None,
            "note": "similarity/rank is appearance evidence only and cannot confirm physical identity",
        },
        "grounding_dino_canary": {
            "status": provider_status["proposal_evidence"]["status"],
            "proposal_count": provider_status["proposal_evidence"]["item_count"],
            "note": "mechanical smoke only; inherited prompt/thresholds are disclosed runtime configuration, not selected or promoted here",
        },
        "bearing_canary": {
            "status": provider_status["bearing_evidence"]["status"],
            "note": "bearing metadata is independent and never upgrades text/visual/proposal evidence",
        },
        "limitations": [
            "Small, intentionally consumed public canary; no fresh, blind, formal, or scientific performance authority.",
            "Wikimedia images are not a representative navigation distribution and carry per-source license/trademark constraints.",
            "OCR string matches, DINOv2 cosine, detector ranking scores, and map bearing remain separate evidence types.",
            "No fusion, evaluator truth, referent decision, temporal identity, navigation action, arrival, safety, Android, or default-App integration.",
        ],
        "next_action": "Use this seam only as a fail-closed provider substrate; any candidate policy or formal cohort requires separate authorization.",
    }
    result_path = evidence_dir / "canary-result.json"
    _write_json(result_path, result)
    closeout_path = evidence_dir / "closeout.md"
    lines = [
        "# NamedReferentProviderV0 canary closeout",
        "",
        f"Claim ceiling: `{CLAIM_CEILING}`.",
        "",
        "This was a public, repeatable, non-fresh platform-engineering canary. It made no scientific performance,",
        "navigation, safety, model-promotion, or default-App claim. No evidence fusion or referent decision ran.",
        "",
        "## Provider status",
        "",
    ]
    for channel, status in provider_status.items():
        lines.append(f"- `{channel}`: `{status['status']}`, {status['item_count']} items, {status['latency_ms']:.2f} ms")
        if status["error"]:
            lines.append(f"  - error: `{status['error']['code']}` — {status['error']['message']}")
    lines.extend(["", "## OCR runtime", ""])
    for device, summary in ocr_by_device.items():
        lines.append(
            f"- `{device}`: {summary['statuses']}; median={summary['latency']['median_ms']} ms; "
            f"polygons={summary['polygon_item_count']}; match items={summary['match_class_item_counts']}"
        )
    lines.extend([
        "",
        "## Reference-image runtime",
        "",
        f"- same-POI candidate ranks: `{target_ranks}` across {len(ranking)} target/distractor queries.",
        "- DINOv2 cosine/ranking is appearance evidence only, not physical-identity confirmation.",
        "",
        "## Limits",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        f"Machine result: `{result_path}`",
        f"Source/license manifest: `{manifest_path}`",
    ])
    closeout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    primary_ocr.close()
    return result_path, closeout_path
