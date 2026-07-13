from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "docs" / "research" / "frontier-upgrade-2026-07"
PDF_DIR = ROOT / ".downloads" / "papers" / "2026-07-frontier-upgrade"
TEXT_DIR = PDF_DIR / "text"
SOURCE_MANIFEST = PROJECT / "refs" / "paper-sources.json"
OUTPUT = PROJECT / "refs" / "paper-inventory.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sources = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = []
    for source in sources:
        pdf_path = PDF_DIR / source["file"]
        reader = PdfReader(str(pdf_path))
        page_text = []
        extracted_chars = 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            extracted_chars += len(text)
            page_text.append(f"\n\n===== PDF PAGE {number} =====\n\n{text}")
        text_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        text_path.write_text("".join(page_text), encoding="utf-8")
        item = dict(source)
        item.update(
            {
                "bytes": pdf_path.stat().st_size,
                "sha256": sha256(pdf_path),
                "pages": len(reader.pages),
                "extracted_chars": extracted_chars,
                "text_file": str(text_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        inventory.append(item)
        print(f"{source['id']}\tpages={item['pages']}\tchars={extracted_chars}\t{source['file']}")
    OUTPUT.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
