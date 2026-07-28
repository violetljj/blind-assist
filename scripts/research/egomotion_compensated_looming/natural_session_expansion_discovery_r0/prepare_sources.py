from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


RECORD_ID = "1476931"
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"
SEALED_SESSION = 16
DISCOVERY_ARCHIVES = {
    13: (137269248, "f838f36f4b6aa3c9a2375a39e7a26128"),
    14: (139189043, "04b1a475813d93b7f198945b588350be"),
    15: (54845329, "f5febcd087acd90531aea98efff71c7c"),
    17: (160543225, "ed86ea48759576323e50a16a44f6cd85"),
}
REQUIRED_MEMBERS = (
    "iphone/frames.mov",
    "iphone/frames.csv",
    "ground-truth/pose.csv",
)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(session: int, archive: Path) -> None:
    if session == SEALED_SESSION:
        raise PermissionError("SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN")
    expected_bytes, expected_md5 = DISCOVERY_ARCHIVES[session]
    if archive.exists():
        if (
            archive.stat().st_size != expected_bytes
            or digest(archive, "md5") != expected_md5
        ):
            raise ValueError(f"EXISTING_ARCHIVE_IDENTITY_MISMATCH:{session}")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".zip.part")
    if temporary.exists():
        raise FileExistsError(f"PARTIAL_ARCHIVE_EXISTS:{temporary}")
    url = f"{BASE_URL}/advio-{session:02d}.zip/content"
    with urllib.request.urlopen(url) as response, temporary.open("xb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    if (
        temporary.stat().st_size != expected_bytes
        or digest(temporary, "md5") != expected_md5
    ):
        raise ValueError(f"DOWNLOADED_ARCHIVE_IDENTITY_MISMATCH:{session}")
    temporary.replace(archive)


def extract(session: int, archive: Path, source_root: Path) -> dict[str, object]:
    if session == SEALED_SESSION:
        raise PermissionError("SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN")
    if source_root.exists():
        raise FileExistsError(f"SOURCE_ROOT_EXISTS:{source_root}")
    prefix = f"advio-{session:02d}/"
    selected = [prefix + relative for relative in REQUIRED_MEMBERS]
    receipt_members: dict[str, dict[str, object]] = {}
    source_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        missing = sorted(set(selected) - names)
        if missing:
            raise ValueError(f"REQUIRED_ARCHIVE_MEMBERS_MISSING:{missing}")
        for member in selected:
            relative = member.removeprefix(prefix)
            destination = source_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            info = bundle.getinfo(member)
            receipt_members[relative] = {
                "archive_member_bytes": info.file_size,
                "archive_member_crc32": f"{info.CRC:08x}",
                "sha256": digest(destination, "sha256"),
            }
    return {
        "schema": "rcle.natural_session.source_receipt.v1",
        "session_number": session,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_md5": digest(archive, "md5"),
        "source_root": source_root.as_posix(),
        "members": receipt_members,
        "sealed_session_touched": False,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def adopt_existing(
    session: int, archive: Path, source_root: Path
) -> dict[str, object]:
    if session == SEALED_SESSION:
        raise PermissionError("SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN")
    expected_bytes, expected_md5 = DISCOVERY_ARCHIVES[session]
    if (
        not archive.is_file()
        or archive.stat().st_size != expected_bytes
        or digest(archive, "md5") != expected_md5
    ):
        raise ValueError("EXISTING_ARCHIVE_IDENTITY_MISMATCH")
    members: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive) as bundle:
        prefix = f"advio-{session:02d}/"
        for relative in REQUIRED_MEMBERS:
            path = source_root / relative
            info = bundle.getinfo(prefix + relative)
            if not path.is_file() or path.stat().st_size != info.file_size:
                raise ValueError(f"EXISTING_SOURCE_MEMBER_MISMATCH:{relative}")
            members[relative] = {
                "archive_member_bytes": info.file_size,
                "archive_member_crc32": f"{info.CRC:08x}",
                "sha256": digest(path, "sha256"),
            }
    return {
        "schema": "rcle.natural_session.source_receipt.v1",
        "session_number": session,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_md5": digest(archive, "md5"),
        "source_root": source_root.resolve().as_posix(),
        "members": members,
        "sealed_session_touched": False,
        "adopted_existing_verified_source": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, choices=(13, 14, 15, 16, 17))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--existing-archive", type=Path)
    parser.add_argument("--existing-source-root", type=Path)
    args = parser.parse_args()
    session = args.session
    if session == SEALED_SESSION:
        raise PermissionError("SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN")
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.existing_archive is not None or args.existing_source_root is not None:
        if (
            args.existing_archive is None
            or args.existing_source_root is None
        ):
            raise ValueError("EXISTING_ARCHIVE_AND_SOURCE_MUST_BE_PAIRED")
        receipt = adopt_existing(
            session,
            args.existing_archive.resolve(),
            args.existing_source_root.resolve(),
        )
        write_json(root / f"source_receipt_advio_{session:02d}.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False))
        return 0
    archive = root / "downloads" / f"advio-{session:02d}.zip"
    source_root = root / "sources" / f"advio-{session:02d}"
    download(session, archive)
    receipt = extract(session, archive, source_root)
    write_json(root / f"source_receipt_advio_{session:02d}.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
