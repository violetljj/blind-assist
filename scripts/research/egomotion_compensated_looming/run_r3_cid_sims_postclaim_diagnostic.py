"""Post-claim topology diagnostic; never represents formal R3 evidence."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
import sys
import zipfile

from scripts.research.egomotion_compensated_looming import r3_cid_sims_geometry_scan as scan


class GroundtruthArchive:
    """Same read fence as R2, with the observed official groundtruth filename."""

    def __init__(self, path: Path) -> None:
        self._archive = zipfile.ZipFile(path)
        infos = self._archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("DIAGNOSTIC_DUPLICATE_MEMBER")
        for info, name in zip(infos, names):
            parsed = PurePosixPath(name)
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or "\\" in name
                or bool(re.match(r"^[A-Za-z]:", name))
                or info.flag_bits & 0x1
            ):
                raise ValueError("DIAGNOSTIC_UNSAFE_MEMBER")
        self.inventory = [
            {
                "name": info.filename,
                "crc32": f"{info.CRC:08x}",
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "is_directory": info.is_dir(),
            }
            for info in infos
        ]
        self._files = {info.filename for info in infos if not info.is_dir()}
        if "floor3_1/groundtruth.txt" not in self._files:
            raise ValueError("DIAGNOSTIC_GROUNDTRUTH_MISSING")
        self.depth_members_read: list[str] = []

    def read_control(self, name: str) -> bytes:
        if name != "pose.txt":
            raise ValueError("DIAGNOSTIC_CONTROL_READ_FORBIDDEN")
        return self._archive.read("floor3_1/groundtruth.txt")

    def read_depth(self, relative: str) -> bytes:
        member = f"floor3_1/{relative}"
        parsed = PurePosixPath(relative)
        if (
            len(parsed.parts) != 2
            or parsed.parts[0] != "depth"
            or member not in self._files
        ):
            raise ValueError("DIAGNOSTIC_DEPTH_MEMBER")
        self.depth_members_read.append(member)
        return self._archive.read(member)

    def member_exists(self, relative: str) -> bool:
        return f"floor3_1/{relative}" in self._files

    def __enter__(self) -> "GroundtruthArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self._archive.close()


def main() -> int:
    scan.geometry.CidSimsArchive = GroundtruthArchive
    exit_code = scan.main()
    output_index = sys.argv.index("--output-dir") + 1
    marker = Path(sys.argv[output_index]) / "DIAGNOSTIC_NOT_FORMAL.txt"
    marker.write_text(
        "Post-claim exploratory diagnostic. The frozen R3 scanner rejected the "
        "archive because it expected pose.txt while the official archive contains "
        "groundtruth.txt. This output may establish development usefulness only; "
        "it is not formal R3 admission or confirmation evidence.\n",
        encoding="utf-8",
        newline="\n",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
