from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "docs" / "research" / "frontier-upgrade-2026-07"
PDF_DIR = ROOT / ".downloads" / "papers" / "2026-07-frontier-upgrade"
INVENTORY = PROJECT / "refs" / "paper-inventory.json"
REPORT = PROJECT / "BLINDASSIST_FRONTIER_PAPER_UPGRADE_REPORT_2026-07.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    papers = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if len(papers) != 14:
        errors.append(f"inventory count={len(papers)}, expected=14")
    total_pages = 0
    report_text = REPORT.read_text(encoding="utf-8")
    for paper in papers:
        pdf = PDF_DIR / paper["file"]
        text_file = ROOT / paper["text_file"]
        if not pdf.is_file():
            errors.append(f"missing PDF: {pdf}")
            continue
        if pdf.read_bytes()[:5] != b"%PDF-":
            errors.append(f"invalid PDF header: {pdf.name}")
        actual_hash = sha256(pdf)
        if actual_hash != paper["sha256"]:
            errors.append(f"hash mismatch: {pdf.name}")
        pages = len(PdfReader(str(pdf)).pages)
        total_pages += pages
        if pages != paper["pages"]:
            errors.append(f"page mismatch: {pdf.name} {pages}!={paper['pages']}")
        if not text_file.is_file():
            errors.append(f"missing text extraction: {text_file}")
        else:
            markers = len(re.findall(r"^===== PDF PAGE \d+ =====$", text_file.read_text(encoding="utf-8"), re.M))
            if markers != pages:
                errors.append(f"page marker mismatch: {pdf.name} {markers}!={pages}")
        if paper["url"] not in report_text:
            errors.append(f"paper not referenced in report: {paper['id']}")
    if total_pages != 162:
        errors.append(f"total pages={total_pages}, expected=162")

    required = [
        PROJECT / "plan" / "project-overview.md",
        PROJECT / "plan" / "outline.md",
        PROJECT / "plan" / "progress.md",
        PROJECT / "refs" / "evidence-map.md",
        PROJECT / "refs" / "paper-inventory.md",
        PROJECT / "notes" / "segmentation.md",
        PROJECT / "notes" / "robustness.md",
        PROJECT / "notes" / "temporal-human.md",
        PROJECT / "notes" / "local-evidence.md",
        REPORT,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {path}")
    if re.search(r"\b(?:TODO|TBD|FIXME)\b|待补|占位", report_text, re.I):
        errors.append("report contains an unresolved placeholder token")
    if "do_not_replace_default_model" not in report_text:
        errors.append("report omits production promotion boundary")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", report_text):
        if re.match(r"^[a-z]+://", target, re.I) or target.startswith("#"):
            continue
        clean = target.split("#", 1)[0]
        linked = (REPORT.parent / clean).resolve()
        if not linked.exists():
            errors.append(f"broken local link: {target}")
    evidence_text = (PROJECT / "refs" / "evidence-map.md").read_text(encoding="utf-8")
    report_sections = set(re.findall(r"^#{2,3}\s+(\d+(?:\.\d+)*)\b", report_text, re.M))
    evidence_slots = set(re.findall(r"\u00a7(\d+(?:\.\d+)*)", evidence_text))
    for section in sorted(evidence_slots - report_sections):
        errors.append(f"evidence map points to missing report section: {section}")

    print(f"papers={len(papers)} total_pages={total_pages} report_chars={len(report_text)}")
    print(f"required_artifacts={len(required)} errors={len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(1)
    print("VERIFICATION_OK")


if __name__ == "__main__":
    main()
