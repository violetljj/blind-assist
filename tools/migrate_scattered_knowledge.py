#!/usr/bin/env python3
"""Migrate the repository's curated legacy research notes into the knowledge reserve.

The importer reads immutable Git revisions plus two hash-recorded local deep-reading
reports.  It never deletes records and keeps one canonical item while adding separate
route-use records for every legacy judgment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = REPO_ROOT / "research" / "knowledge"
ARCHIVE_REVISION = "f01a0072160f9b14c0debaf912bd67a4efb52772"
CURRENT_SOURCE_REVISION = "1aec08abfc193fc5d868fb1e8274613886751f00"
MIGRATION_DATE = "2026-08-28"
MIGRATION_ID = "migration-scattered-knowledge-2026-08-28"

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
HEADING_PATTERN = re.compile(r"^###\s+(\d+)\.\s+(.+)$", re.MULTILINE)
ARXIV_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
DOI_PATTERN = re.compile(r"(?:doi\.org/|^doi:)(10\.[^?#\s]+)", re.IGNORECASE)


@dataclass
class LegacyEntry:
    group_id: str
    legacy_id: str
    route: str
    title: str
    canonical_ref: str
    kind: str
    summary: str
    mechanism: str
    application: str
    limitation: str
    source_ref: str
    evidence_kind: str
    links: list[tuple[str, str]] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    use_state: str = "candidate"
    adoption_mode: str = "mechanism_adaptation"
    tags: list[str] = field(default_factory=list)
    expected_effect: str = ""
    modifications: str = (
        "Only the documented mechanism and evidence boundary were migrated; no "
        "implementation, threshold, route authority, or promotion state changed."
    )


@dataclass
class SourceGroup:
    identifier: str
    source_ref: str
    source_sha256: str
    entries: list[LegacyEntry]


def _run_git_show_bytes(revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git show failed for {revision}:{path}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _git_source(revision: str, path: str) -> tuple[str, str, str]:
    raw = _run_git_show_bytes(revision, path)
    return (
        raw.decode("utf-8"),
        f"{revision}:{path}",
        hashlib.sha256(raw).hexdigest(),
    )


def _artifact_source(path: str) -> tuple[str, str, str]:
    absolute = REPO_ROOT / path
    raw = absolute.read_bytes()
    return raw.decode("utf-8"), path.replace("\\", "/"), hashlib.sha256(raw).hexdigest()


def _clean_markdown(value: str, *, limit: int = 4000) -> str:
    value = LINK_PATTERN.sub(lambda match: match.group(1), value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("**", "").replace("`", "")
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip(" |\n\r\t")
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    return value


def _extract_links(value: str) -> list[tuple[str, str]]:
    return [(_clean_markdown(label, limit=200), url.strip()) for label, url in LINK_PATTERN.findall(value)]


def _canonicalize_reference(reference: str) -> str:
    reference = html.unescape(reference.strip()).rstrip("/.,;")
    arxiv = ARXIV_PATTERN.search(reference)
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv.group(1)}"
    doi = DOI_PATTERN.search(reference)
    if doi:
        return f"https://doi.org/{doi.group(1).rstrip('/').lower()}"
    return reference


def _canonical_key(reference: str) -> str:
    return _canonicalize_reference(reference).casefold().rstrip("/")


def _manual_identity_key(title: str) -> str | None:
    folded = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
    if folded.startswith("closing the gap"):
        return "manual:closing-the-gap-assets-2019"
    if folded.startswith("ai guide dog"):
        return "manual:ai-guide-dog-2025"
    if "project guideline" in folded:
        return "manual:project-guideline"
    return None


def _slug(value: str, *, limit: int = 72) -> str:
    value = value.casefold().replace("++", "-plus-plus")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if not value:
        value = "legacy"
    return value[:limit].rstrip("-")


def _year_from(value: str) -> int | None:
    years = [int(match) for match in re.findall(r"\b(19\d{2}|20\d{2})\b", value)]
    return years[-1] if years else None


def _year_from_reference(reference: str) -> int | None:
    arxiv = ARXIV_PATTERN.search(reference)
    if arxiv:
        prefix = int(arxiv.group(1)[:2])
        return 2000 + prefix if prefix <= 50 else 1900 + prefix
    venue_year = re.search(
        r"(?:CVPR|ICCV|WACV|ECCV|IROS|ICRA|NeurIPS|AAAI)[^0-9]{0,3}(20\d{2})",
        reference,
        re.IGNORECASE,
    )
    return int(venue_year.group(1)) if venue_year else None


def _table_rows(text: str, start: str, end: str | None = None) -> list[list[str]]:
    start_index = text.index(start)
    segment = text[start_index:]
    if end and end in segment:
        segment = segment[: segment.index(end)]
    rows: list[list[str]] = []
    for line in segment.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("|---"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


def _section(text: str, heading: str, next_heading_pattern: str = r"^## ") -> str:
    match = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing section heading: {heading}")
    tail = text[match.end() :]
    next_match = re.search(next_heading_pattern, tail, re.MULTILINE)
    return tail[: next_match.start()] if next_match else tail


def _labeled_value(section: str, labels: Iterable[str], fallback: str = "") -> str:
    for label in labels:
        pattern = re.compile(
            rf"(?:\d+\.\s*)?\*\*[^*]*{re.escape(label)}[^*]*\*\*[：。]?\s*"
            rf"(.*?)(?=\n\s*\n(?:\d+\.\s*)?\*\*|\n---|\Z)",
            re.DOTALL,
        )
        match = pattern.search(section)
        if match:
            return _clean_markdown(match.group(1))
        bullet_match = re.search(
            rf"^-\s*{re.escape(label)}[：:]\s*(.*?)(?=\n-\s*[^\n：:]+[：:]|\n###|\n---|\Z)",
            section,
            re.MULTILINE | re.DOTALL,
        )
        if bullet_match:
            return _clean_markdown(bullet_match.group(1))
    return _clean_markdown(fallback)


def _choose_reference(links: list[tuple[str, str]]) -> str:
    if not links:
        raise RuntimeError("legacy entry has no external link")
    for _, url in links:
        if DOI_PATTERN.search(url):
            return _canonicalize_reference(url)
    return _canonicalize_reference(links[0][1])


def _infer_kind(title: str, url: str, hint: str = "") -> str:
    folded = f"{title} {url} {hint}".casefold()
    if "工具/数据" in hint or any(
        token in folded
        for token in (
            "jrdb family",
            "riskbench",
            "spring + robustspring",
            "aria digital twin",
            "egotraj",
            "dynamicstereo",
            "pie + scenario",
            "revel",
        )
    ):
        return "dataset"
    if "apps.apple.com" in folded or "smartais" in folded or "running guide agent" in folded:
        return "project"
    if any(token in folded for token in ("official repository", "github.com/google-research/project-guideline", "soundscape")):
        return "tool"
    if any(token in folded for token in ("crtp", "heads-up")):
        return "project"
    return "paper"


def _entry(
    *,
    group: str,
    alias: str,
    route: str,
    title: str,
    links: list[tuple[str, str]],
    summary: str,
    mechanism: str,
    application: str,
    limitation: str,
    source_ref: str,
    evidence_kind: str,
    kind: str | None = None,
    year: int | None = None,
    venue: str = "",
    state: str = "candidate",
    mode: str = "mechanism_adaptation",
    tags: list[str] | None = None,
    expected_effect: str = "",
) -> LegacyEntry:
    if not links:
        raise RuntimeError(f"{alias}: no external links")
    canonical = _choose_reference(links)
    clean_title = _clean_markdown(title, limit=500)
    clean_summary = _clean_markdown(summary) or f"Legacy research note for {clean_title}."
    clean_mechanism = _clean_markdown(mechanism) or clean_summary
    clean_application = _clean_markdown(application) or clean_mechanism
    clean_limitation = _clean_markdown(limitation) or (
        "The legacy note did not establish a route-level result, deployment claim, "
        "user outcome, or safety authority."
    )
    return LegacyEntry(
        group_id=group,
        legacy_id=alias,
        route=route,
        title=clean_title,
        canonical_ref=canonical,
        kind=kind or _infer_kind(clean_title, canonical),
        summary=clean_summary,
        mechanism=clean_mechanism,
        application=clean_application,
        limitation=clean_limitation,
        source_ref=source_ref,
        evidence_kind=evidence_kind,
        links=[(label, _canonicalize_reference(url)) for label, url in links if _canonicalize_reference(url) != canonical],
        year=(
            year
            if year is not None
            else _year_from(f"{title} {summary}") or _year_from_reference(canonical)
        ),
        venue=venue,
        use_state=state,
        adoption_mode=mode,
        tags=tags or [],
        expected_effect=_clean_markdown(expected_effect) if expected_effect else clean_application,
    )


def _load_dtr_01_30() -> SourceGroup:
    path = "idea.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    rows = [row for row in _table_rows(text, "### 动态出行风险：30 项外部候选池") if row and re.fullmatch(r"DR\d{2}", row[0])]
    if [row[0] for row in rows] != [f"DR{index:02d}" for index in range(1, 31)]:
        raise RuntimeError("DR01-DR30 source is incomplete or reordered")
    entries: list[LegacyEntry] = []
    for row in rows:
        legacy, source_cell, problem, mechanism, evidence_domain, limitation = row[:6]
        links = _extract_links(source_cell)
        split_links = links if legacy == "DR01" else [links[0]]
        for link_index, selected in enumerate(split_links):
            suffix = ".ab"[link_index + 1] if legacy == "DR01" else ""
            alias = f"dtr:{legacy}{suffix}"
            title = selected[0] if legacy == "DR01" else links[0][0]
            extra_links = [selected] + ([] if legacy == "DR01" else links[1:])
            entries.append(
                _entry(
                    group="dtr-01-30",
                    alias=alias,
                    route="dtr-r0",
                    title=title,
                    links=extra_links,
                    summary=problem,
                    mechanism=mechanism,
                    application=mechanism,
                    limitation=limitation,
                    source_ref=source_ref,
                    evidence_kind="git",
                    kind=_infer_kind(title, selected[1], evidence_domain),
                    tags=["dynamic-travel-risk", "legacy-candidate"],
                )
            )
    if len(entries) != 31:
        raise RuntimeError(f"expected 31 DR01-DR30 mappings, got {len(entries)}")
    return SourceGroup("dtr-01-30", source_ref, digest, entries)


def _load_dtr_31_40() -> SourceGroup:
    path = "docs/research/dynamic-travel-risk/DYNAMIC_TRAVEL_RISK_ADDITIONAL_10_DEEP_READING_2026-08-24.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    matches = list(re.finditer(r"^## (DR(?:3[1-9]|40)) — (.+)$", text, re.MULTILINE))
    if [match.group(1) for match in matches] != [f"DR{index}" for index in range(31, 41)]:
        raise RuntimeError("DR31-DR40 sections are incomplete or reordered")
    entries: list[LegacyEntry] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end() : end]
        legacy = match.group(1)
        links = _extract_links(_labeled_value_raw(section, "一手来源"))
        entries.append(
            _entry(
                group="dtr-31-40",
                alias=f"dtr:{legacy}",
                route="dtr-r0",
                title=match.group(2).split("：", 1)[0],
                links=links,
                summary=_labeled_value(section, ["它真正问的问题"]),
                mechanism=_labeled_value(section, ["核心机制"]),
                application=_labeled_value(section, ["取舍", "核心价值"]),
                limitation=_labeled_value(section, ["最重要的证据边界", "证据边界"]),
                source_ref=source_ref,
                evidence_kind="git",
                tags=["dynamic-travel-risk", "deep-reading"],
            )
        )
    return SourceGroup("dtr-31-40", source_ref, digest, entries)


def _labeled_value_raw(section: str, label: str) -> str:
    match = re.search(
        rf"\*\*{re.escape(label)}[。.]?\*\*\s*(.*?)(?=\n\s*\n\*\*|\n---|\Z)",
        section,
        re.DOTALL,
    )
    return match.group(1) if match else ""


def _load_dtr_41_60() -> SourceGroup:
    path = "research/active/dtr-r0/LITERATURE_RESERVE_2026-08-27.md"
    text, source_ref, digest = _git_source(CURRENT_SOURCE_REVISION, path)
    rows = [row for row in _table_rows(text, "#") if row and re.fullmatch(r"DR(?:4[1-9]|5\d|60)", row[0])]
    if [row[0] for row in rows] != [f"DR{index}" for index in range(41, 61)]:
        raise RuntimeError("DR41-DR60 source is incomplete or reordered")
    entries = [
        _entry(
            group="dtr-41-60",
            alias=f"dtr:{row[0]}",
            route="dtr-r0",
            title=_extract_links(row[1])[0][0],
            links=_extract_links(row[1]),
            summary=row[2],
            mechanism=row[2],
            application=row[3],
            limitation=row[4],
            source_ref=source_ref,
            evidence_kind="git",
            tags=["dynamic-travel-risk", "literature-reserve"],
        )
        for row in rows
    ]
    return SourceGroup("dtr-41-60", source_ref, digest, entries)


def _load_l10() -> SourceGroup:
    path = "research/active/l10-r0/LITERATURE_RESERVE_20_2026-08-27.md"
    text, source_ref, digest = _git_source(CURRENT_SOURCE_REVISION, path)
    rows = [row for row in _table_rows(text, "#") if row and re.fullmatch(r"\d{1,2}", row[0])]
    if [int(row[0]) for row in rows] != list(range(1, 21)):
        raise RuntimeError("L10 1-20 source is incomplete or reordered")
    entries = [
        _entry(
            group="l10-01-20",
            alias=f"l10:{row[0]}",
            route="l10-r0",
            title=_extract_links(row[1])[0][0],
            links=_extract_links(row[1]),
            summary=row[2],
            mechanism=row[2],
            application=row[3],
            limitation=row[4],
            source_ref=source_ref,
            evidence_kind="git",
            tags=["l10", "literature-reserve"],
        )
        for row in rows
    ]
    return SourceGroup("l10-01-20", source_ref, digest, entries)


def _load_ustrf() -> SourceGroup:
    path = "docs/research/ustrf-sc/USTRF_FRONTIER_PAPER_GUIDE_2026-07-22.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    rows = [row for row in _table_rows(text, "#") if row and re.fullmatch(r"P\d{2}", row[0])]
    if [row[0] for row in rows] != [f"P{index:02d}" for index in range(1, 14)]:
        raise RuntimeError("USTRF P01-P13 source is incomplete or reordered")
    entries: list[LegacyEntry] = []
    for row in rows:
        links = _extract_links(row[1])
        external_links = [(label, url) for label, url in links if url.startswith(("https://", "http://"))]
        title_match = re.search(r"\[([^\]]+)\]\(", row[1])
        title = _clean_markdown(title_match.group(1), limit=500) if title_match else links[0][0]
        entries.append(
            _entry(
                group="ustrf-p01-p13",
                alias=f"ustrf:{row[0]}",
                route="ustrf-sc",
                title=title,
                links=external_links,
                summary=row[2],
                mechanism=row[2],
                application=row[3],
                limitation=row[5],
                source_ref=source_ref,
                evidence_kind="git",
                state="retired",
                mode="reference",
                tags=["ustrf", "frontier-guide"],
            )
        )
    return SourceGroup("ustrf-p01-p13", source_ref, digest, entries)


def _load_goal_prior() -> SourceGroup:
    path = "docs/research/goal-copilot/P0_PRIOR_ART_ASSIMILATION_2026-08-21.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    rows = [row for row in _table_rows(text, "## 逐项吸收", "## BlindAssist 模块映射") if row and _extract_links(row[0])]
    if len(rows) != 9:
        raise RuntimeError(f"expected 9 Goal prior-art rows, got {len(rows)}")
    split_rows = {3, 4, 6}
    entries: list[LegacyEntry] = []
    for row_index, row in enumerate(rows, start=1):
        links = _extract_links(row[0])
        selected_links = links if row_index in split_rows else [links[0]]
        for link_index, selected in enumerate(selected_links):
            suffix = chr(ord("a") + link_index) if len(selected_links) > 1 else ""
            alias = f"goal-prior:P0-R{row_index:02d}{suffix}"
            entries.append(
                _entry(
                    group="goal-prior-art",
                    alias=alias,
                    route="goal-copilot-p0",
                    title=selected[0],
                    links=[selected] + ([] if len(selected_links) > 1 else links[1:]),
                    summary=row[1],
                    mechanism=row[1],
                    application=row[2],
                    limitation=row[3],
                    source_ref=source_ref,
                    evidence_kind="git",
                    kind=_infer_kind(selected[0], selected[1]),
                    state="adopted",
                    mode="reference",
                    tags=["goal-copilot", "prior-art"],
                )
            )
    if len(entries) != 12:
        raise RuntimeError(f"expected 12 Goal prior-art mappings, got {len(entries)}")
    return SourceGroup("goal-prior-art", source_ref, digest, entries)


def _load_rcle() -> SourceGroup:
    path = "docs/research/rcle/RCLE_PRIOR_MECHANISM_EVIDENCE_MAP_R0_2026-07-28.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    rows = [row for row in _table_rows(text, "## 证据表", "## 对当前 RCLE 的决策") if row and re.match(r"E[1-5] ", row[0])]
    if len(rows) != 5:
        raise RuntimeError(f"expected 5 RCLE rows, got {len(rows)}")
    entries: list[LegacyEntry] = []
    for index, row in enumerate(rows, start=1):
        links = _extract_links(row[0])
        title_match = re.search(r"\*([^*]+)\*", row[0])
        title = title_match.group(1) if title_match else links[0][0]
        entries.append(
            _entry(
                group="rcle-e1-e5",
                alias=f"rcle:E{index}",
                route="rcle",
                title=title,
                links=links,
                summary=row[2],
                mechanism=row[2],
                application=row[5],
                limitation=row[6],
                source_ref=source_ref,
                evidence_kind="git",
                state="adopted" if index <= 2 else "retired",
                mode="reference",
                tags=["rcle", "motion", "related-work"],
            )
        )
    return SourceGroup("rcle-e1-e5", source_ref, digest, entries)


def _deep_reading_sections(text: str, expected: range) -> list[tuple[int, str, str]]:
    matches = [match for match in HEADING_PATTERN.finditer(text) if int(match.group(1)) in expected]
    if [int(match.group(1)) for match in matches] != list(expected):
        raise RuntimeError(f"deep-reading sections are incomplete for {expected.start}-{expected.stop - 1}")
    result: list[tuple[int, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append((int(match.group(1)), match.group(2), text[match.end() : end]))
    return result


def _load_goal_deep_reading(path: str, expected: range, group_id: str) -> SourceGroup:
    text, source_ref, digest = _artifact_source(path)
    entries: list[LegacyEntry] = []
    for number, heading_title, section in _deep_reading_sections(text, expected):
        links = _extract_links(section)
        citation = _labeled_value(section, ["精确引文与链接", "引文"])
        citation_links = _extract_links(_labeled_value_raw_any(section, ["精确引文与链接", "引文"]))
        selected_links = citation_links or links
        title = heading_title
        verdict = _labeled_value(section, ["判定"])
        if not verdict:
            verdict_match = re.search(r"判定[：:]\s*`?([A-Z_]+)`?", section)
            verdict = verdict_match.group(1) if verdict_match else ""
        verdict_upper = verdict.upper()
        state = "candidate" if "DIRECT_COMPONENT" in verdict_upper else "adopted"
        mode = "mechanism_adaptation" if "DIRECT_COMPONENT" in verdict_upper else "reference"
        if "DEFER" in verdict_upper:
            state = "retired"
        entries.append(
            _entry(
                group=group_id,
                alias=f"goal-audit:{number:02d}",
                route="goal-copilot-p0",
                title=title,
                links=selected_links,
                summary=_labeled_value(section, ["实际证明", "证明与未证明", "问题与 I/O", "问题 / I-O", "原文"]),
                mechanism=_labeled_value(section, ["核心机制", "机制"]),
                application=_labeled_value(section, ["BlindAssist 映射", "对当前失败的映射", "对 BlindAssist"]),
                limitation=_labeled_value(section, ["未证明", "证明与未证明"]),
                expected_effect=_labeled_value(section, ["最小实验", "最小可证伪适配"]),
                source_ref=source_ref,
                evidence_kind="artifact",
                state=state,
                mode=mode,
                tags=["goal-copilot", "deep-reading", f"paper-{number:02d}"],
            )
        )
        if not citation and not _labeled_value(section, ["原文"]):
            raise RuntimeError(f"goal-audit:{number:02d}: missing citation text")
    return SourceGroup(group_id, source_ref, digest, entries)


def _labeled_value_raw_any(section: str, labels: Iterable[str]) -> str:
    for label in labels:
        match = re.search(
            rf"(?:\d+\.\s*)?\*\*[^*]*{re.escape(label)}[^*]*\*\*[：。]?\s*"
            rf"(.*?)(?=\n\s*\n(?:\d+\.\s*)?\*\*|\n---|\Z)",
            section,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        bullet_match = re.search(
            rf"^-\s*{re.escape(label)}[：:]\s*(.*?)(?=\n-\s*[^\n：:]+[：:]|\n###|\n---|\Z)",
            section,
            re.MULTILINE | re.DOTALL,
        )
        if bullet_match:
            return bullet_match.group(1)
    return ""


def _load_frontier_upgrade() -> SourceGroup:
    inventory_path = "docs/research/frontier-upgrade-2026-07/refs/paper-inventory.json"
    report_path = "docs/research/frontier-upgrade-2026-07/BLINDASSIST_FRONTIER_PAPER_UPGRADE_REPORT_2026-07.md"
    inventory_text, inventory_ref, inventory_digest = _git_source(ARCHIVE_REVISION, inventory_path)
    report_text, report_ref, report_digest = _git_source(ARCHIVE_REVISION, report_path)
    inventory = json.loads(inventory_text)
    group_headings = {
        "PID": "3.1 PIDNet 与 Mobile-Seed：边界必须参与融合，但边界监督可能反伤主任务",
        "MSEED": "3.1 PIDNet 与 Mobile-Seed：边界必须参与融合，但边界监督可能反伤主任务",
        "MRFP": "3.2 MRFP、UPC、SWSEG：三个模块解决三种不同失败，不能一次叠加",
        "UPC": "3.2 MRFP、UPC、SWSEG：三个模块解决三种不同失败，不能一次叠加",
        "SWSEG": "3.2 MRFP、UPC、SWSEG：三个模块解决三种不同失败，不能一次叠加",
        "VALUES": "3.3 ValUES 与 Kandinsky：先回答“不确定性有没有用”，再谈校准",
        "KAND": "3.3 ValUES 与 Kandinsky：先回答“不确定性有没有用”，再谈校准",
        "DTERN": "3.4 DTERN、BOFP 与 Escalator Problem：时序目标不是平滑，而是有效事件",
        "BOFP": "3.4 DTERN、BOFP 与 Escalator Problem：时序目标不是平滑，而是有效事件",
        "ESCALATOR": "3.4 DTERN、BOFP 与 Escalator Problem：时序目标不是平滑，而是有效事件",
        "STEPP": "3.5 Watch Your STEPP：轨迹监督可以教“熟悉的可走区域”，但异常不等于危险",
        "AIGD": "3.6 AI Guide Dog、VisAssist 与 CLIP-BLV：真实用户分布不能用模拟或通用数据代替",
        "VISASSIST": "3.6 AI Guide Dog、VisAssist 与 CLIP-BLV：真实用户分布不能用模拟或通用数据代替",
        "CLIPBLV": "3.6 AI Guide Dog、VisAssist 与 CLIP-BLV：真实用户分布不能用模拟或通用数据代替",
    }
    entries: list[LegacyEntry] = []
    for paper in inventory:
        heading = group_headings[paper["id"]]
        analysis = _section(report_text, f"### {heading}", r"^### |^## ")
        analysis_clean = _clean_markdown(analysis, limit=2600)
        entries.append(
            _entry(
                group="frontier-upgrade-14",
                alias=f"frontier:{paper['id']}",
                route="frontier-upgrade-2026-07",
                title=paper["title"],
                links=[("Primary source", paper["url"])],
                summary=analysis_clean,
                mechanism=analysis_clean,
                application=f"Imported under the report's {heading} mechanism group for bounded project comparison.",
                limitation="The frontier-upgrade report treats this as paper-derived mechanism evidence. It does not independently establish BlindAssist event utility, mobile deployment, BLV user benefit, model promotion, or safety.",
                source_ref=report_ref,
                evidence_kind="git",
                kind="paper",
                year=paper.get("year"),
                venue=paper.get("venue", ""),
                state="adopted",
                mode="reference",
                tags=["frontier-upgrade", "paper-review"],
            )
        )
    combined_digest = hashlib.sha256(f"{inventory_digest}:{report_digest}".encode()).hexdigest()
    return SourceGroup(
        "frontier-upgrade-14",
        f"{inventory_ref} + {report_ref}",
        combined_digest,
        entries,
    )


def _load_hftf() -> SourceGroup:
    path = "docs/research/hftf/HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    # The historical charter has one three-column linked-neighbour table.
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.strip().startswith("| [") and len(line.strip().strip("|").split("|")) == 3
    ]
    entries: list[LegacyEntry] = []
    for row_index, row in enumerate(rows, start=1):
        links = _extract_links(row[0])
        for link_index, selected in enumerate(links):
            suffix = chr(ord("a") + link_index) if len(links) > 1 else ""
            entries.append(
                _entry(
                    group="hftf-neighbours",
                    alias=f"hftf:H{row_index:02d}{suffix}",
                    route="hftf",
                    title=selected[0],
                    links=[selected],
                    summary=row[1],
                    mechanism=row[1],
                    application=row[2],
                    limitation=f"The charter records this as an adjacent mechanism, not HFTF evidence. {row[2]}",
                    source_ref=source_ref,
                    evidence_kind="git",
                    state="adopted",
                    mode="reference",
                    tags=["hftf", "adjacent-work"],
                )
            )
    if len(entries) != 9:
        raise RuntimeError(f"expected 9 HFTF work mappings, got {len(entries)}")
    return SourceGroup("hftf-neighbours", source_ref, digest, entries)


def _load_taro() -> SourceGroup:
    path = "docs/research/taro/TARO_R0_RESEARCH_ROUTE_GUIDE_2026-08-10.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    section = _section(text, "### 3.3 主要相邻文献与论文空位", r"^## |^### ")
    bullets = [match.group(1) for match in re.finditer(r"^- (.*?)(?=^- |\Z)", section, re.MULTILINE | re.DOTALL)]
    entries: list[LegacyEntry] = []
    for bullet_index, bullet in enumerate(bullets, start=1):
        links = _extract_links(bullet)
        if not links:
            continue
        clean_bullet = _clean_markdown(bullet)
        for link_index, selected in enumerate(links):
            suffix = chr(ord("a") + link_index) if len(links) > 1 else ""
            entries.append(
                _entry(
                    group="taro-adjacent-work",
                    alias=f"taro:T{bullet_index:02d}{suffix}",
                    route="taro-r0",
                    title=selected[0],
                    links=[selected],
                    summary=clean_bullet,
                    mechanism=clean_bullet,
                    application=clean_bullet,
                    limitation="This adjacent work constrains TARO novelty or supplies a bounded reference. Its source-domain result is not TARO task-functional identifiability, a metric anchor, O0R truth, human-constrained sensing evidence, user benefit, or safety.",
                    source_ref=source_ref,
                    evidence_kind="git",
                    state="adopted",
                    mode="reference",
                    tags=["taro", "adjacent-work"],
                )
            )
    if len(entries) != 16:
        raise RuntimeError(f"expected 16 TARO work mappings, got {len(entries)}")
    return SourceGroup("taro-adjacent-work", source_ref, digest, entries)


def _load_project_guideline_audit() -> SourceGroup:
    path = "docs/PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    matches = list(
        re.finditer(
            r"^### 3\.(\d) (.+?) — `?(ADAPT|REFERENCE|HOLD|DROP)`?\s*$",
            text,
            re.MULTILINE,
        )
    )
    if [int(match.group(1)) for match in matches] != list(range(1, 9)):
        raise RuntimeError("Project Guideline component audit is incomplete")
    slugs = [
        "stop-failure-semantics",
        "logging-fields",
        "replay-recomputation",
        "occupancy-map",
        "pose-time-alignment",
        "depth-scale-alignment",
        "audio-feedback",
        "unreal-pretrained-models",
    ]
    entries: list[LegacyEntry] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else text.find("\n## 4.", match.end())
        section = text[match.end() : end if end >= 0 else len(text)]
        decision = match.group(3)
        state = {
            "ADAPT": "adopted",
            "REFERENCE": "adopted",
            "HOLD": "planned",
            "DROP": "rejected",
        }[decision]
        mode = "mechanism_adaptation" if decision == "ADAPT" else "reference"
        description = _clean_markdown(section, limit=3000)
        entries.append(
            _entry(
                group="project-guideline-components",
                alias=f"project-guideline:{slugs[index]}",
                route="project-guideline-component-audit",
                title="Google Research Project Guideline",
                links=[
                    (
                        "Fixed audited revision",
                        "https://github.com/google-research/project-guideline/tree/b5fa173de36ab591d875492a899358cdc5843291",
                    )
                ],
                summary=f"Component audit decision for {match.group(2)}: {decision}.",
                mechanism=description,
                application=description,
                limitation=(
                    f"The component decision is {decision}. The audit absorbs only the stated "
                    "engineering boundary; it does not import the full C++/Bazel/MediaPipe/ARCore/"
                    "Unreal stack or establish dynamic-risk, grounding, user, product, or safety evidence."
                ),
                source_ref=source_ref,
                evidence_kind="git",
                kind="tool",
                state=state,
                mode=mode,
                tags=["project-guideline", "component-audit", decision.casefold()],
            )
        )
    return SourceGroup("project-guideline-components", source_ref, digest, entries)


def _load_idea_explicit_sources() -> SourceGroup:
    path = "docs/history/idea/IDEA_ARCHIVE_THROUGH_2026-07-28.md"
    _, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    rows = [
        {
            "alias": "idea:SANPO",
            "route": "sanpo",
            "title": "SANPO: A Scene Understanding, Accessibility and Human Navigation Dataset",
            "links": [
                ("Official dataset", "https://google-research-datasets.github.io/sanpo_dataset/"),
                ("WACV 2025 paper", "https://openaccess.thecvf.com/content/WACV2025/papers/Waghmare_SANPO_A_Scene_Understanding_Accessibility_and_Human_Navigation_Dataset_WACV_2025_paper.pdf"),
            ],
            "kind": "dataset",
            "mechanism": "Use scene-understanding and accessibility masks as continuous corridor, boundary, obstacle, and unknown evidence with source-aware sequence structure.",
            "application": "Support SANPO segmentation and corridor experiments while keeping source masks below event-lifecycle truth and independent promotion gates.",
            "limitation": "SANPO source masks are not BlindAssist event onset/clearance truth, collision outcome, user benefit, mobile model promotion, or safety evidence.",
            "mode": "dataset",
        },
        {
            "alias": "idea:MobileNetV3",
            "route": "sanpo",
            "title": "Searching for MobileNetV3",
            "links": [("ICCV 2019 paper", "https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf")],
            "kind": "algorithm",
            "mechanism": "MobileNetV3Small plus Lite R-ASPP provides a mobile-oriented lightweight segmentation backbone/decoder candidate.",
            "application": "Retain as a bounded four-class SANPO segmentation candidate only after event and data gates close; compare on event metrics and device latency rather than paper accuracy alone.",
            "limitation": "The paper does not establish BlindAssist corridor semantics, event utility, INT8 fidelity on the target phone, BLV user benefit, or safety.",
            "mode": "component",
        },
        {
            "alias": "idea:Depth-Anything-V2",
            "route": "sanpo",
            "title": "Depth Anything V2",
            "links": [("arXiv", "https://arxiv.org/abs/2406.09414")],
            "kind": "paper",
            "mechanism": "Generalizable monocular relative depth used as an offline teacher or bounded confirmation/abstention evidence source.",
            "application": "Keep relative-depth ablations separate from metric distance and prevent depth alone from promoting an alert.",
            "limitation": "Relative monocular depth is not metric range, route truth, event truth, user evidence, or safety authority; historical conversion/runtime limitations remain separate use evidence.",
            "mode": "component",
        },
        {
            "alias": "idea:RT-DETR",
            "route": "detector-lab",
            "title": "RT-DETR official project",
            "links": [("Official repository", "https://github.com/lyuwenyu/RT-DETR")],
            "kind": "project",
            "mechanism": "Real-time transformer detector family retained only as a detector-adapter comparison candidate.",
            "application": "Compare through the same proposal and event contract; public AP or speed cannot independently authorize a default detector change.",
            "limitation": "The project does not establish BlindAssist event recall, critical-miss behavior, phone latency, BLV benefit, or safety.",
            "mode": "component",
        },
        {
            "alias": "idea:SAM2",
            "route": "offline-annotation",
            "title": "SAM 2: Segment Anything in Images and Videos",
            "links": [("arXiv", "https://arxiv.org/abs/2408.00714")],
            "kind": "paper",
            "mechanism": "Prompted image/video mask propagation retained as an offline annotation, occlusion, teacher, or upper-bound mechanism.",
            "application": "Use only behind explicit prompts or frozen offline protocols; never let an unprompted mask directly trigger a risk alert.",
            "limitation": "SAM 2 masks do not create target identity, route relevance, event truth, user outcome, real-time phone feasibility, or safety authority.",
            "mode": "evaluator",
        },
    ]
    entries = [
        _entry(
            group="idea-explicit-sources",
            alias=row["alias"],
            route=row["route"],
            title=row["title"],
            links=row["links"],
            summary=row["mechanism"],
            mechanism=row["mechanism"],
            application=row["application"],
            limitation=row["limitation"],
            source_ref=source_ref,
            evidence_kind="git",
            kind=row["kind"],
            state="candidate",
            mode=row["mode"],
            tags=["idea-archive", row["route"]],
        )
        for row in rows
    ]
    return SourceGroup("idea-explicit-sources", source_ref, digest, entries)


def _load_idea_synthesis() -> SourceGroup:
    path = "docs/history/idea/IDEA_ARCHIVE_THROUGH_2026-07-28.md"
    text, source_ref, digest = _git_source(ARCHIVE_REVISION, path)
    section = _section(text, "### 2026-07-13：文献驱动的研究指导（记录）", r"^## |^### ")
    paragraphs = [line[2:] for line in section.splitlines() if line.startswith("- ")]
    summary = " ".join(paragraphs)
    synthetic_url = f"git:{source_ref}"
    entry = LegacyEntry(
        group_id="idea-assistive-navigation-synthesis",
        legacy_id="idea:assistive-navigation-2026-07",
        route="sanpo",
        title="Assistive-navigation literature synthesis through 2026-07-13",
        canonical_ref=synthetic_url,
        kind="survey",
        summary=_clean_markdown(summary, limit=3000),
        mechanism=_clean_markdown(" ".join(paragraphs[1:5]), limit=3000),
        application=_clean_markdown(" ".join(paragraphs[1:7]), limit=3000),
        limitation="The archived synthesis names a local reading cache but does not retain enough bibliographic identity for every unnamed paper. It is migrated as one survey record rather than fabricating individual canonical sources. Its guidance is not model, event, user, or safety evidence.",
        source_ref=source_ref,
        evidence_kind="git",
        year=2026,
        use_state="adopted",
        adoption_mode="reference",
        tags=["sanpo", "assistive-navigation", "literature-synthesis"],
        expected_effect="Keep the event-lifecycle, corridor, uncertainty, and model-order guidance searchable without inventing missing paper identities.",
    )
    return SourceGroup("idea-assistive-navigation-synthesis", source_ref, digest, [entry])


def load_source_groups() -> list[SourceGroup]:
    groups = [
        _load_dtr_01_30(),
        _load_dtr_31_40(),
        _load_dtr_41_60(),
        _load_l10(),
        _load_ustrf(),
        _load_goal_prior(),
        _load_project_guideline_audit(),
        _load_rcle(),
        _load_goal_deep_reading(
            "artifacts.local/research/exa-results/blindassist-30-papers-deep-reading-notes-2026-08-24.md",
            range(1, 31),
            "goal-deep-reading-01-30",
        ),
        _load_goal_deep_reading(
            "artifacts.local/research/exa-results/blindassist-mainline-additional-10-deep-reading-2026-08-24.md",
            range(31, 41),
            "goal-deep-reading-31-40",
        ),
        _load_frontier_upgrade(),
        _load_hftf(),
        _load_taro(),
        _load_idea_explicit_sources(),
        _load_idea_synthesis(),
    ]
    identifiers = [group.identifier for group in groups]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("duplicate source-group id")
    return groups


def _load_json_records(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _new_item_id(entry: LegacyEntry, items: dict[str, dict[str, Any]]) -> str:
    prefix = entry.kind if entry.kind in {"paper", "algorithm", "project", "dataset", "tool", "survey"} else "item"
    title_slug = _slug(entry.title)
    if not re.search(r"[a-z]", title_slug):
        title_slug = f"{title_slug}-{_slug(entry.legacy_id)}"
    base = f"{prefix}-{title_slug}"
    candidate = base
    index = 2
    while candidate in items:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _find_identity_item(
    entry: LegacyEntry,
    items: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    canonical: dict[str, str],
    manual: dict[str, str],
) -> str | None:
    if entry.legacy_id in aliases:
        return aliases[entry.legacy_id]
    canonical_owner = canonical.get(_canonical_key(entry.canonical_ref))
    if canonical_owner:
        return canonical_owner
    manual_key = _manual_identity_key(entry.title)
    return manual.get(manual_key) if manual_key else None


def _mechanism_id(entry: LegacyEntry) -> str:
    return _slug(entry.legacy_id.replace(":", "-"), limit=90)


def _use_id(entry: LegacyEntry, uses: dict[str, dict[str, Any]]) -> str:
    base = f"use-{_slug(entry.route, limit=38)}-{_slug(entry.legacy_id, limit=56)}"
    candidate = base
    index = 2
    while candidate in uses:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _evidence(entry: LegacyEntry) -> dict[str, str]:
    return {
        "kind": entry.evidence_kind,
        "ref": entry.source_ref,
        "summary": f"Legacy source for {entry.legacy_id}; imported without changing its recorded evidence boundary.",
    }


def build_library(groups: list[SourceGroup]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    items = _load_json_records(KNOWLEDGE_ROOT / "items")
    uses = _load_json_records(KNOWLEDGE_ROOT / "uses")
    aliases = {
        alias: item_id
        for item_id, item in items.items()
        for alias in item.get("aliases", [])
    }
    canonical = {
        _canonical_key(item["canonical_ref"]): item_id for item_id, item in items.items()
    }
    manual = {
        key: item_id
        for item_id, item in items.items()
        if (key := _manual_identity_key(item.get("title", "")))
    }
    manifest_groups: list[dict[str, Any]] = []

    for group in groups:
        mappings: list[dict[str, Any]] = []
        for entry in group.entries:
            owner = _find_identity_item(entry, items, aliases, canonical, manual)
            disposition = "deduplicated" if owner else "migrated"
            if owner is None:
                owner = _new_item_id(entry, items)
                items[owner] = {
                    "schema_version": 1,
                    "id": owner,
                    "kind": entry.kind,
                    "title": entry.title,
                    "canonical_ref": entry.canonical_ref,
                    "authors": [],
                    "year": entry.year,
                    "venue": entry.venue,
                    "summary": entry.summary,
                    "mechanisms": [],
                    "tags": [],
                    "aliases": [],
                    "links": [],
                    "added_at": MIGRATION_DATE,
                    "updated_at": MIGRATION_DATE,
                }
                canonical[_canonical_key(entry.canonical_ref)] = owner
                manual_key = _manual_identity_key(entry.title)
                if manual_key:
                    manual[manual_key] = owner

            item = items[owner]
            if item.get("year") is None and entry.year is not None:
                item["year"] = entry.year
            if not item.get("venue") and entry.venue:
                item["venue"] = entry.venue
            if entry.legacy_id not in item["aliases"]:
                item["aliases"].append(entry.legacy_id)
                aliases[entry.legacy_id] = owner
            mechanism_id = _mechanism_id(entry)
            existing_mechanisms = {mechanism["id"] for mechanism in item["mechanisms"]}
            if mechanism_id not in existing_mechanisms:
                item["mechanisms"].append(
                    {
                        "id": mechanism_id,
                        "name": f"{entry.title} — {entry.legacy_id} route mechanism",
                        "description": entry.mechanism,
                        "inputs": [],
                        "outputs": [],
                        "limitations": entry.limitation,
                    }
                )
            item["tags"] = _unique(item.get("tags", []) + entry.tags + [entry.route])
            known_refs = {link["ref"] for link in item.get("links", [])}
            for label, reference in entry.links:
                if reference != item["canonical_ref"] and reference not in known_refs:
                    item["links"].append({"label": label, "ref": reference})
                    known_refs.add(reference)
            if entry.canonical_ref != item["canonical_ref"] and entry.canonical_ref not in known_refs:
                item["links"].append(
                    {"label": f"Canonical source used by {entry.legacy_id}", "ref": entry.canonical_ref}
                )
            item["updated_at"] = MIGRATION_DATE

            existing_use_ids = [
                use_id
                for use_id, use in uses.items()
                if use.get("item_id") == owner
                and use.get("route") == entry.route
                and mechanism_id in use.get("mechanism_ids", [])
            ]
            if not existing_use_ids:
                # Existing seed records use a hand-authored mechanism id. Resolve them by alias+route.
                if disposition == "deduplicated" and entry.legacy_id in {"l10:1", "dtr:DR45"}:
                    existing_use_ids = [
                        use_id
                        for use_id, use in uses.items()
                        if use.get("item_id") == owner and use.get("route") == entry.route
                    ]
                    if existing_use_ids:
                        # The rich seed already represents this legacy judgment; remove the generic duplicate mechanism.
                        item["mechanisms"] = [
                            mechanism
                            for mechanism in item["mechanisms"]
                            if mechanism["id"] != mechanism_id
                        ]
                if not existing_use_ids:
                    use_id = _use_id(entry, uses)
                    uses[use_id] = {
                        "schema_version": 1,
                        "id": use_id,
                        "item_id": owner,
                        "route": entry.route,
                        "mechanism_ids": [mechanism_id],
                        "use_state": entry.use_state,
                        "adoption_mode": entry.adoption_mode,
                        "usage": {
                            "source_scope": entry.mechanism,
                            "project_application": entry.application,
                            "modifications": entry.modifications,
                            "expected_effect": entry.expected_effect,
                        },
                        "evaluation": {
                            "reproduction_status": "not_attempted",
                            "verdict": "not_run",
                            "setup": "",
                            "effect": "",
                            "metrics": [],
                            "claim_boundary": entry.limitation,
                        },
                        "evidence": [_evidence(entry)],
                        "history": [
                            {
                                "date": MIGRATION_DATE,
                                "change": f"Migrated {entry.legacy_id} from the scattered legacy knowledge source; no execution authority was created.",
                            }
                        ],
                        "added_at": MIGRATION_DATE,
                        "updated_at": MIGRATION_DATE,
                    }
                    existing_use_ids = [use_id]

            # Keep migration-owned evidence pinned to the exact source commit even
            # when this importer is rerun after a tag or branch moves.
            legacy_summary_prefix = f"Legacy source for {entry.legacy_id};"
            for use_id in existing_use_ids:
                for evidence in uses[use_id].get("evidence", []):
                    if evidence.get("summary", "").startswith(legacy_summary_prefix):
                        evidence.update(_evidence(entry))

            mappings.append(
                {
                    "legacy_id": entry.legacy_id,
                    "item_id": owner,
                    "use_ids": sorted(existing_use_ids),
                    "disposition": "synthesis" if entry.kind == "survey" else disposition,
                    "note": (
                        "Mapped to one canonical item and retained as an independent route-use judgment."
                        if disposition == "migrated"
                        else "Deduplicated by canonical identity while retaining this legacy alias and route-use judgment."
                    ),
                }
            )
        manifest_groups.append(
            {
                "id": group.identifier,
                "source_ref": group.source_ref,
                "source_sha256": group.source_sha256,
                "expected_entries": len(group.entries),
                "mappings": mappings,
            }
        )

    _apply_result_overrides(items, uses, manifest_groups)
    manifest = {
        "schema_version": 1,
        "id": MIGRATION_ID,
        "created_at": MIGRATION_DATE,
        "scope": (
            "Curated literature, algorithm-mechanism, dataset, tool, and exemplary-project "
            "notes that previously lived in route documents, archive-tag documents, or "
            "hash-recorded local deep-reading reports."
        ),
        "source_groups": manifest_groups,
        "exclusions": [
            {
                "scope": "Incidental citations in implementation, status, protocol, and result documents",
                "reason": "They are evidence pointers rather than deliberately curated reusable knowledge records; result documents remain linked from use evidence when they change a mechanism verdict.",
            },
            {
                "scope": "Unnamed papers in the 2026-07 assistive-navigation cache synthesis",
                "reason": "The archived note does not retain enough canonical bibliographic identity for every unnamed source, so the synthesis is preserved as one survey item instead of fabricating identities.",
            },
            {
                "scope": "PDFs, repositories, checkpoints, and raw reproduction artifacts",
                "reason": "The knowledge reserve stores metadata and evidence references; payloads remain under artifacts.local or their original external locations.",
            },
        ],
    }
    return items, uses, manifest


def _mapping_for_alias(manifest_groups: list[dict[str, Any]], alias: str) -> dict[str, Any]:
    for group in manifest_groups:
        for mapping in group["mappings"]:
            if mapping["legacy_id"] == alias:
                return mapping
    raise RuntimeError(f"missing migration mapping for {alias}")


def _append_history(use: dict[str, Any], change: str) -> None:
    if not any(event.get("change") == change for event in use["history"]):
        use["history"].append({"date": MIGRATION_DATE, "change": change})
    use["updated_at"] = MIGRATION_DATE


def _add_evidence(use: dict[str, Any], evidence: dict[str, str]) -> None:
    if not any(
        existing.get("kind") == evidence["kind"] and existing.get("ref") == evidence["ref"]
        for existing in use["evidence"]
    ):
        use["evidence"].append(evidence)


def _replace_git_path_evidence(
    use: dict[str, Any], path: str, summary: str
) -> None:
    use["evidence"] = [
        evidence
        for evidence in use["evidence"]
        if not (
            evidence.get("kind") == "git"
            and evidence.get("ref", "").endswith(f":{path}")
        )
    ]
    _add_evidence(
        use,
        {
            "kind": "git",
            "ref": f"{ARCHIVE_REVISION}:{path}",
            "summary": summary,
        },
    )


def _apply_result_overrides(
    items: dict[str, dict[str, Any]],
    uses: dict[str, dict[str, Any]],
    manifest_groups: list[dict[str, Any]],
) -> None:
    nearid_mapping = _mapping_for_alias(manifest_groups, "goal-audit:31")
    nearid_use = uses[nearid_mapping["use_ids"][0]]
    nearid_use["use_state"] = "rejected"
    nearid_use["evaluation"].update(
        {
            "reproduction_status": "partial",
            "verdict": "negative",
            "setup": "A frozen NearID-style DINOv2-S projection arm on a source-disjoint CORe50 protocol; the official NearID SigLIP2/MAP checkpoint was not run.",
            "effect": "The arm rescued 4 baseline errors but caused 17 collateral regressions, so this specific adaptation was sealed and rejected.",
            "metrics": [
                "rescue=4",
                "collateral=17",
                "control_retention=5.6%",
                "same_instance_recall=3.7%",
            ],
            "claim_boundary": "This rejects only the frozen DINOv2-S projection-head arm and its protocol. It does not reproduce or falsify the official NearID architecture, every hard-negative method, physical-instance identity, NONE calibration, user benefit, or safety.",
        }
    )
    _replace_git_path_evidence(
        nearid_use,
        "docs/research/goal-copilot/NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0_RESULT_2026-08-24.md",
        "Frozen NearID-style unary arm result and sealed negative verdict.",
    )
    _append_history(
        nearid_use,
        "Attached the later frozen NearID-style unary result: negative with severe collateral; official NearID remained untested.",
    )

    layout_mapping = _mapping_for_alias(manifest_groups, "goal-audit:37")
    layout_use = uses[layout_mapping["use_ids"][0]]
    layout_use["use_state"] = "rejected"
    layout_use["evaluation"].update(
        {
            "reproduction_status": "partial",
            "verdict": "mixed",
            "setup": "An analytic reciprocal spatial-layout adaptation on 900 source-disjoint Washington RGB-D object pairs; it was not a reproduction of the published Doppelgangers model.",
            "effect": "Spatial layout produced real rescues and perfect invariance but substantially underperformed the DINO baseline and introduced excessive collateral, closing this passive arm.",
            "metrics": [
                "dino_mean_nearest=702/900",
                "analytic_reciprocal_layout=558/900",
                "rescue=74",
                "collateral=218",
                "control_retention=68.9%",
                "stable_distractor=29/69",
                "invariance=100%",
            ],
            "claim_boundary": "This rejects the frozen analytic reciprocal-layout adaptation on this protocol, not the Doppelgangers publication, every learned layout verifier, multi-view evidence, or physical-instance identity in general.",
        }
    )
    _replace_git_path_evidence(
        layout_use,
        "docs/research/goal-copilot/SPATIAL_LAYOUT_IDENTITY_VERIFICATION_V0_RESULT_2026-08-24.md",
        "Frozen analytic spatial-layout arm with mixed signal and unacceptable collateral.",
    )
    _append_history(
        layout_use,
        "Attached the later analytic spatial-layout result: mixed signal, excessive collateral, and closed passive arm.",
    )

    abot_mapping = _mapping_for_alias(manifest_groups, "goal-prior:P0-R08")
    abot_item = items[abot_mapping["item_id"]]
    mechanism_id = _mechanism_id(
        LegacyEntry(
            "", "goal-prior:P0-R08", "", "", "", "paper", "", "", "", "", "", ""
        )
    )
    reproduction_use_id = "use-goal-copilot-p2-abotn-official-waypoint"
    if reproduction_use_id not in uses:
        uses[reproduction_use_id] = {
            "schema_version": 1,
            "id": reproduction_use_id,
            "item_id": abot_item["id"],
            "route": "goal-copilot-p2",
            "mechanism_ids": [mechanism_id],
            "use_state": "adopted",
            "adoption_mode": "direct_replication",
            "usage": {
                "source_scope": "The official ABotN waypoint interface and native metric-arrival evaluation only.",
                "project_application": "Run the official waypoint stack to separate metric approach from final visual handoff and user completion.",
                "modifications": "BlindAssist preserved a separate visual-handoff/completion gate instead of treating native metric arrival as destination completion.",
                "expected_effect": "Establish whether the official waypoint mechanism can reach its native metric threshold while exposing the remaining handoff boundary.",
            },
            "evaluation": {
                "reproduction_status": "partial",
                "verdict": "mixed",
                "setup": "Official ABotN waypoint runs on named trajectories with native/oracle arrival plus an independent BlindAssist visual-handoff check.",
                "effect": "One run reached native metric arrival at 0.9246 m, while visual handoff and completion remained false; other trajectories exhausted budget or failed earlier grounding.",
                "metrics": [
                    "traj_3_distance=12.5335m->2.9600m",
                    "traj_5_final_distance=0.9246m",
                    "traj_5_native_arrival=true",
                    "traj_5_visual_handoff=false",
                ],
                "claim_boundary": "This establishes one task's official native metric arrival only. It does not establish public referent grounding, entrance affordance, visual handoff, user completion, BLV benefit, live-device control, generalization, or safety.",
            },
            "evidence": [
                {
                    "kind": "git",
                    "ref": f"{ARCHIVE_REVISION}:docs/research/goal-copilot/BLINDASSIST_ABOTN_OFFICIAL_WAYPOINT_HANDOFF_V1_RESULT_2026-08-24.md",
                    "summary": "Official waypoint run with metric-arrival versus visual-handoff separation.",
                }
            ],
            "history": [
                {
                    "date": MIGRATION_DATE,
                    "change": "Migrated the official ABotN waypoint reproduction result and retained its metric-arrival-only claim ceiling.",
                }
            ],
            "added_at": MIGRATION_DATE,
            "updated_at": MIGRATION_DATE,
        }
    if reproduction_use_id not in abot_mapping["use_ids"]:
        abot_mapping["use_ids"].append(reproduction_use_id)
    _replace_git_path_evidence(
        uses[reproduction_use_id],
        "docs/research/goal-copilot/BLINDASSIST_ABOTN_OFFICIAL_WAYPOINT_HANDOFF_V1_RESULT_2026-08-24.md",
        "Official waypoint run with metric-arrival versus visual-handoff separation.",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_serialized(payload))


def _read_text_exact(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _serialized(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_or_check(
    items: dict[str, dict[str, Any]],
    uses: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    *,
    check: bool,
) -> int:
    targets = {
        **{KNOWLEDGE_ROOT / "items" / f"{identifier}.json": payload for identifier, payload in items.items()},
        **{KNOWLEDGE_ROOT / "uses" / f"{identifier}.json": payload for identifier, payload in uses.items()},
        KNOWLEDGE_ROOT / "migrations" / f"{MIGRATION_ID}.json": manifest,
    }
    mismatches: list[Path] = []
    for path, payload in sorted(targets.items(), key=lambda pair: str(pair[0])):
        desired = _serialized(payload)
        if not path.is_file() or _read_text_exact(path) != desired:
            mismatches.append(path)
            if not check:
                _write_json(path, payload)
    if check and mismatches:
        print("Migration output is stale or missing:", file=sys.stderr)
        for path in mismatches:
            print(f" - {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    mapping_count = sum(len(group["mappings"]) for group in manifest["source_groups"])
    action = "PASS" if check else "WROTE"
    print(
        f"{action} scattered-knowledge migration: items={len(items)} uses={len(uses)} "
        f"source_groups={len(manifest['source_groups'])} legacy_mappings={mapping_count}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write missing or changed migration outputs")
    mode.add_argument("--check", action="store_true", help="verify outputs match the frozen sources")
    args = parser.parse_args(argv)
    try:
        groups = load_source_groups()
        items, uses, manifest = build_library(groups)
        return write_or_check(items, uses, manifest, check=args.check)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
