"""Prepare training-user negative-only support controls without NumPy or GPU.

Only metadata is decoded. Packed query features are hashed in one streaming
pass and referenced, never loaded or copied. Plan boxes/masks are skipped by
the selective JSON reader. Existing sampled query rows and folds are fixed.
"""
from __future__ import annotations

import argparse
import ast
import codecs
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import struct
import tarfile
import time
import zipfile


CHUNK = 1024 * 1024
JSON_STRING_TOKEN = re.compile(r'["\\]')
JSON_STRUCT_TOKEN = re.compile(r'["{}\[\]]')


def source_hash(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(Path(path).resolve()), "bytes": size, "sha256": digest.hexdigest()}


def read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated NPY input")
    return data


def npy_header(stream):
    if read_exact(stream, 6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, minor = read_exact(stream, 2)
    if major == 1:
        length = struct.unpack("<H", read_exact(stream, 2))[0]
        offset = 10
    elif major in (2, 3):
        length = struct.unpack("<I", read_exact(stream, 4))[0]
        offset = 12
    else:
        raise ValueError(f"unsupported NPY version {(major, minor)}")
    if length > CHUNK:
        raise ValueError("unexpectedly large NPY header")
    header = ast.literal_eval(read_exact(stream, length).decode("utf-8" if major == 3 else "latin1"))
    if not isinstance(header, dict) or header.get("fortran_order") is not False:
        raise ValueError("only C-order NPY arrays are supported")
    shape, dtype = header.get("shape"), header.get("descr")
    if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid NPY shape")
    if not isinstance(dtype, str) or not re.fullmatch(r"[<>|=][Uiubf][0-9]+", dtype):
        raise ValueError(f"unsupported NPY dtype {dtype!r}; no object/pickle arrays permitted")
    width = int(dtype[2:]) * (4 if dtype[1] == "U" else 1)
    return {"shape": list(shape), "dtype": dtype, "itemsize": width,
            "header_bytes": offset + length, "payload_bytes": math.prod(shape) * width}


def decode_npy(stream):
    header = npy_header(stream)
    count, dtype, width = math.prod(header["shape"]), header["dtype"], header["itemsize"]
    if header["payload_bytes"] > 32 * CHUNK:
        raise ValueError("metadata array too large; refusing feature-array decode")
    payload = read_exact(stream, header["payload_bytes"])
    if stream.read(1):
        raise ValueError("trailing bytes in NPY member")
    if dtype[1] == "U":
        encoding = "utf-32-be" if dtype[0] == ">" else "utf-32-le"
        values = [payload[i * width:(i + 1) * width].decode(encoding).rstrip("\x00") for i in range(count)]
    elif dtype[1] in "iu":
        order = "big" if dtype[0] == ">" else "little"
        values = [int.from_bytes(payload[i * width:(i + 1) * width], order, signed=dtype[1] == "i")
                  for i in range(count)]
    elif dtype in ("|b1", "<b1", ">b1", "=b1"):
        values = [bool(value) for value in payload]
    else:
        raise ValueError(f"not a supported metadata type: {dtype}")
    return header, values


def npz_member(archive, name, header_only=False):
    with archive.open(name + ".npy") as member:
        return npy_header(member) if header_only else decode_npy(member)


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


class SelectiveJSON:
    """Bounded-buffer JSON field reader; ignored subtrees are never decoded."""
    def __init__(self, stream):
        self.stream = stream
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer, self.pos, self.eof = "", 0, False

    def more(self):
        if self.eof:
            return False
        raw = self.stream.read(CHUNK)
        self.buffer = self.buffer[self.pos:] + self.decoder.decode(raw, final=not raw)
        self.pos = 0
        self.eof = not raw
        return bool(raw)

    def char(self):
        while self.pos >= len(self.buffer):
            if not self.more() and self.pos >= len(self.buffer):
                return ""
        return self.buffer[self.pos]

    def whitespace(self):
        while self.char() and self.buffer[self.pos].isspace():
            self.pos += 1

    def expect(self, char):
        self.whitespace()
        if self.char() != char:
            raise ValueError(f"expected JSON {char!r}")
        self.pos += 1

    def string(self):
        self.expect('"')
        self.pos -= 1
        while True:
            try:
                value, end = json.decoder.scanstring(self.buffer, self.pos + 1)
                self.pos = end
                return value
            except json.JSONDecodeError:
                if not self.more():
                    raise ValueError("unterminated JSON string") from None

    def skip_string(self):
        self.expect('"')
        while True:
            match = JSON_STRING_TOKEN.search(self.buffer, self.pos)
            if match is None:
                self.pos = len(self.buffer)
                if not self.more():
                    raise ValueError("unterminated skipped JSON string")
                continue
            self.pos = match.end()
            if match.group() == '"':
                return
            # Ignore the escaped code unit without interpreting its value.
            if not self.char():
                raise ValueError("truncated JSON escape")
            self.pos += 1

    def skip(self):
        self.whitespace()
        first = self.char()
        if first == '"':
            self.skip_string()
            return
        if first in ("{", "["):
            stack = ["}" if first == "{" else "]"]
            self.pos += 1
            while stack:
                match = JSON_STRUCT_TOKEN.search(self.buffer, self.pos)
                if match is None:
                    self.pos = len(self.buffer)
                    if not self.more():
                        raise ValueError("unterminated skipped JSON container")
                    continue
                self.pos = match.start()
                token = self.buffer[self.pos]
                if token == '"':
                    self.skip_string()
                else:
                    self.pos += 1
                    if token in "{[":
                        stack.append("}" if token == "{" else "]")
                    elif token != stack.pop():
                        raise ValueError("mismatched skipped JSON container")
            return
        if not first:
            raise ValueError("missing JSON value")
        while self.char() and self.char() not in ",}] \t\r\n":
            self.pos += 1

    def strings_or_null(self):
        self.whitespace()
        if self.char() != "[":
            self.skip()
            return None
        self.expect("[")
        out = []
        self.whitespace()
        while self.char() != "]":
            out.append(self.string())
            self.whitespace()
            if self.char() == "]":
                break
            self.expect(",")
        self.expect("]")
        return out

    def video(self):
        self.expect("{")
        video = {}
        self.whitespace()
        while self.char() != "}":
            key = self.string()
            self.expect(":")
            if key in ("user", "obj", "type", "video"):
                video[key] = self.string()
            elif key == "context":
                video[key] = self.strings_or_null()
            else:
                self.skip()
            self.whitespace()
            if self.char() == "}":
                break
            self.expect(",")
        self.expect("}")
        if any(key not in video for key in ("user", "obj", "type", "video")):
            raise ValueError("plan video lacks required identity fields")
        return video

    def plan(self):
        self.expect("{")
        videos = None
        self.whitespace()
        while self.char() != "}":
            key = self.string()
            self.expect(":")
            if key == "videos":
                if videos is not None:
                    raise ValueError("duplicate plan videos field")
                videos = []
                self.expect("[")
                self.whitespace()
                while self.char() != "]":
                    videos.append(self.video())
                    self.whitespace()
                    if self.char() == "]":
                        break
                    self.expect(",")
                self.expect("]")
            else:
                self.skip()
            self.whitespace()
            if self.char() == "}":
                break
            self.expect(",")
        self.expect("}")
        self.whitespace()
        if self.char():
            raise ValueError("trailing plan JSON content")
        if not videos:
            raise ValueError("empty/missing plan videos")
        return videos


def read_plan(path):
    with Path(path).open("rb") as stream:
        source = HashReader(stream)
        videos = SelectiveJSON(source).plan()
    return videos, {"path": str(Path(path).resolve()), "bytes": source.bytes,
                    "sha256": source.digest.hexdigest(),
                    "decoded_fields": ["videos.user", "videos.obj", "videos.type", "videos.video", "videos.context"],
                    "skipped_without_decoding": ["boxes", "mask", "valid", "spans", "singles", "other fields"]}


def frame_number(name):
    try:
        return int(PurePosixPath(name.replace("\\", "/")).name.rsplit("-", 1)[1].split(".")[0])
    except (IndexError, ValueError):
        raise ValueError(f"unrecognized frame number in {name!r}") from None


def validate_name(name, video=None):
    if not isinstance(name, str) or not name or "\n" in name or "\r" in name:
        raise ValueError("invalid frame name")
    windows, posix = PureWindowsPath(name), PurePosixPath(name.replace("\\", "/"))
    if windows.drive or posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"frame is not a safe relative name: {name!r}")
    if video is not None and posix.name.rsplit("-", 1)[0] != video:
        raise ValueError(f"frame/video identity mismatch: {name!r} vs {video!r}")
    if not 1 <= frame_number(name) <= 1000:
        raise ValueError(f"cached frame lies outside the original <=1000 cap: {name!r}")


def build_manifest(videos, packed_meta, patch_dir, fold_of):
    by_video = {}
    support = {}
    for video in videos:
        vid, user, obj = video["video"], video["user"], video["obj"]
        if vid in by_video:
            raise ValueError(f"duplicate plan video {vid}")
        by_video[vid] = video
        if video["type"] == "clean" and user in fold_of:
            context = video.get("context")
            if context is None:
                raise ValueError(f"missing context list for {vid}")
            selected = [name for name in context if frame_number(name) <= 1000]
            for name in selected:
                validate_name(name, vid)
            support.setdefault(user, {}).setdefault(obj, []).extend(selected)
    if set(support) != set(fold_of):
        raise ValueError("fold user set differs from plan clean-video user set")
    users = [{"user": user, "fold": fold_of[user],
              "objects": [{"object": obj, "context": support[user][obj]} for obj in sorted(support[user])],
              "queries": []} for user in sorted(fold_of)]
    if any(len(u["objects"]) < 2 or any(not obj["context"] for obj in u["objects"]) for u in users):
        raise ValueError("empty support object or fewer than two objects")
    by_user = {u["user"]: u for u in users}
    with zipfile.ZipFile(packed_meta) as archive:
        meta = {key: npz_member(archive, key) for key in ("user", "video", "y", "valid")}
    n_rows = len(meta["user"][1])
    if any(meta[key][0]["shape"] != [n_rows] for key in ("user", "video", "y")):
        raise ValueError("packed identity arrays must be aligned row vectors")
    valid_header, valid = meta["valid"]
    if len(valid_header["shape"]) != 2 or valid_header["shape"][0] != n_rows:
        raise ValueError("invalid packed valid-slot shape")
    max_objects = valid_header["shape"][1]
    files = sorted(p for p in Path(patch_dir).glob("*.npz") if not p.name.startswith("_"))
    offset, receipts, names_seen = 0, [], set()
    for path in files:
        receipts.append(source_hash(path))
        with zipfile.ZipFile(path) as archive:
            names_header, names = npz_member(archive, "names")
            user_header, patch_user = npz_member(archive, "user")
            y_header, patch_y = npz_member(archive, "y")
            score_header = npz_member(archive, "S", header_only=True)
        if names_header["shape"] != [len(names)] or user_header["shape"] != [] or y_header["shape"] != []:
            raise ValueError(f"unexpected patch metadata shape in {path.name}")
        if len(names) != 24 or len(set(names)) != len(names):
            raise ValueError(f"expected 24 distinct cached query names in {path.name}")
        if offset + len(names) > n_rows or path.stem not in by_video:
            raise ValueError(f"patch file absent from plan or beyond packed rows: {path.name}")
        info = by_video[path.stem]
        user, obj = info["user"], info["obj"]
        if info["type"] != "clutter" or user not in by_user or patch_user != [user]:
            raise ValueError(f"patch/plan user or type mismatch in {path.name}")
        objects = [entry["object"] for entry in by_user[user]["objects"]]
        if obj not in objects or patch_y != [objects.index(obj)]:
            raise ValueError(f"patch/plan sorted-object label mismatch in {path.name}")
        if score_header["shape"] != [len(names), 256, len(objects)]:
            raise ValueError(f"patch S shape does not match objects/names in {path.name}")
        expected_valid = [True] * len(objects) + [False] * (max_objects - len(objects))
        for row, name in enumerate(names, start=offset):
            validate_name(name, path.stem)
            if name in names_seen:
                raise ValueError(f"duplicate query frame across videos: {name}")
            names_seen.add(name)
            if (meta["user"][1][row] != user or meta["video"][1][row] != path.stem or
                    meta["y"][1][row] != objects.index(obj) or
                    valid[row * max_objects:(row + 1) * max_objects] != expected_valid):
                raise ValueError(f"packed/source metadata mismatch at global query row {row}")
        by_user[user]["queries"].append({"video": path.stem, "object": obj, "frames": names,
                                         "cache_rows": list(range(offset, offset + len(names)))})
        offset += len(names)
    if offset != n_rows or any(not u["queries"] for u in users):
        raise ValueError("packed rows not exactly exhausted or user has no cached queries")
    if len(users) != 44 or len(files) != 461 or n_rows != 11064:
        raise ValueError(f"wrong training cohort: {len(users)} users/{len(files)} videos/{n_rows} rows")
    return users, n_rows, receipts


def feature_metadata(path, n_rows):
    with Path(path).open("rb") as stream:
        source = HashReader(stream)
        header = npy_header(source)
        for _ in iter(lambda: source.read(CHUNK), b""):
            pass
    if header["shape"] != [n_rows, 256, 384] or header["dtype"] != "<f2":
        raise ValueError(f"unexpected query feature layout: {header}")
    if source.bytes != header["header_bytes"] + header["payload_bytes"]:
        raise ValueError("feature NPY size does not match header")
    return header, {"path": str(Path(path).resolve()), "bytes": source.bytes,
                    "sha256": source.digest.hexdigest(), "payload_decoded": False, "copied": False}


class HashWriter:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("short tar output write")
        self.digest.update(data)
        self.bytes += written
        return written

    def flush(self):
        self.stream.flush()


def pack_context(users, frames_root, tar_path, list_path):
    names, seen = [], set()
    for user in users:
        for obj in user["objects"]:
            for name in obj["context"]:
                if name not in seen:
                    names.append(name)
                    seen.add(name)
    frames_root = Path(frames_root).resolve(strict=True)
    paths = []
    for name in names:
        validate_name(name)
        source = (frames_root / name).resolve(strict=True)
        if not source.is_relative_to(frames_root) or not source.is_file():
            raise ValueError(f"context frame escapes root or is not a file: {name}")
        paths.append(source)
    with Path(list_path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(name + "\n" for name in names))
    files = []
    with Path(tar_path).open("xb") as stream:
        output = HashWriter(stream)
        with tarfile.open(fileobj=output, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, path in zip(names, paths):
                stat = path.stat()
                info = tarfile.TarInfo(name.replace("\\", "/"))
                info.size, info.mode, info.mtime = stat.st_size, 0o644, 0
                with path.open("rb") as raw:
                    source = HashReader(raw)
                    archive.addfile(info, source)
                    if source.bytes != stat.st_size or raw.read(1):
                        raise ValueError(f"context file changed during packaging: {name}")
                files.append({"name": name, "bytes": source.bytes, "sha256": source.digest.hexdigest()})
    return {"frames": len(names), "source_bytes": sum(f["bytes"] for f in files),
            "tar": {"path": str(Path(tar_path).resolve()), "bytes": output.bytes,
                    "sha256": output.digest.hexdigest()},
            "file_list": source_hash(list_path), "files": files}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fixture_npy(dtype, shape, payload):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape})
    encoded = (header + " " * ((-(10 + len(header) + 1)) % 64) + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded + payload


def self_test():
    header, strings = decode_npy(io.BytesIO(fixture_npy("<U3", (2,), "ab\0中\0\0".encode("utf-32-le"))))
    assert header["shape"] == [2] and strings == ["ab", "中"]
    _, integers = decode_npy(io.BytesIO(fixture_npy("<i8", (2,), struct.pack("<qq", 3, -2))))
    assert integers == [3, -2]
    _, boolean = decode_npy(io.BytesIO(fixture_npy("|b1", (2, 2), bytes([1, 0, 0, 1]))))
    assert boolean == [True, False, False, True]
    _, scalar = decode_npy(io.BytesIO(fixture_npy("<i8", (), struct.pack("<q", 7))))
    assert scalar == [7]
    sample = {"spans": [{"unused": [1, {"escaped": "a\\\"[b]"}]}], "videos": [
        {"user": "P1", "obj": "cup", "type": "clean", "video": "P1--cup--clean--v",
         "boxes": {"ignore": [1, 2]}, "context": ["P1--cup--clean--v-00001.jpg"]},
        {"user": "P1", "obj": "cup", "type": "clutter", "video": "P1--cup--clutter--v",
         "context": None, "boxes": {"ignore": [1, 2]}}]}
    class TinyReads(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 17) if size >= 0 else 17)
    raw = json.dumps(sample, ensure_ascii=False).encode("utf-8")
    reader = HashReader(TinyReads(raw))
    selected = SelectiveJSON(reader).plan()
    assert len(selected) == 2 and "boxes" not in selected[0] and selected[1]["context"] is None
    assert reader.digest.hexdigest() == hashlib.sha256(raw).hexdigest()
    validate_name("P1--cup--clean--v-00001.jpg", "P1--cup--clean--v")
    print("SELF_TEST_PASS: stdlib NPY unicode/int/bool/scalar and tiny-chunk selective JSON skipping")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "frames", "packed-meta", "query-features", "patch-dir", "folds-json", "out-dir"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    required = ("plan", "frames", "packed_meta", "query_features", "patch_dir", "folds_json", "out_dir")
    if any(getattr(args, key) is None for key in required):
        parser.error("all seven input/output path arguments are required")
    started = time.perf_counter()
    outputs = {name: args.out_dir / name for name in ("manifest.json", "context_file_list.txt", "context_frames.tar", "receipt.json")}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("output already exists; use a fresh --out-dir to preserve prior evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_bytes = args.folds_json.read_bytes()
    folds = json.loads(fold_bytes)["fold_of_user"]
    if (len(folds) != 44 or any(type(v) is not int or v not in range(4) for v in folds.values()) or
            any(list(folds.values()).count(fold) != 11 for fold in range(4))):
        raise ValueError("expected existing 44-user/four-fold assignment, 11 users in every fold")
    videos, plan_receipt = read_plan(args.plan)
    users, n_rows, patch_receipts = build_manifest(videos, args.packed_meta, args.patch_dir, folds)
    header, feature_receipt = feature_metadata(args.query_features, n_rows)
    metadata_receipt = source_hash(args.packed_meta)
    manifest = {
        "schema": 1, "role": "training_user_Development_replay_negative_only_holdout",
        "query_cache": {"n_rows": n_rows, "shape": header["shape"], "dtype": header["dtype"],
                        "features_sha256": feature_receipt["sha256"], "metadata_sha256": metadata_receipt["sha256"]},
        "fold_of_user": folds, "users": users,
        "fixed_split": {"main_holdout_fold": 0, "development_folds": [1, 2, 3],
                        "report_all_folds": True, "retune_on_holdout": False},
        "evidence_identity": "Previously consumed ORBIT train Development; fixed cached queries, not fresh confirmation"}
    write_json(outputs["manifest.json"], manifest)
    context = pack_context(users, args.frames, outputs["context_frames.tar"], outputs["context_file_list.txt"])
    receipt = {
        "schema": 1, "status": "COMPLETE", "users": len(users),
        "query_videos": sum(len(user["queries"]) for user in users), "query_frames": n_rows,
        "context_references": sum(len(obj["context"]) for user in users for obj in user["objects"]),
        "context_unique_frames": context["frames"],
        "fold0_users": sorted(user for user, fold in folds.items() if fold == 0),
        "development_users": sorted(user for user, fold in folds.items() if fold != 0),
        "fold_counts": {str(fold): sum(value == fold for value in folds.values()) for fold in range(4)},
        "sources": {"plan": plan_receipt, "packed_meta": metadata_receipt,
                    "query_features": feature_receipt,
                    "folds_json": {"path": str(args.folds_json.resolve()), "bytes": len(fold_bytes),
                                   "sha256": hashlib.sha256(fold_bytes).hexdigest()},
                    "patch_npz": patch_receipts, "script": source_hash(__file__)},
        "outputs": {"manifest": source_hash(outputs["manifest.json"]), "context": context},
        "validation": {"packed_order": "sorted source video NPZs, all contiguous global rows matched",
                       "labels": "patch scalar y and every packed y match sorted clean-support object identities",
                       "valid_slots": "every packed valid row matches exactly the enrolled sorted object slots",
                       "query_names": "24 unique names/video, frame/video identity, cap, global uniqueness checked",
                       "context_order": "plan videos order within each sorted object; only frame numbers <=1000",
                       "features_loaded_or_copied": False, "boxes_or_masks_decoded": False},
        "wall_seconds": time.perf_counter() - started}
    write_json(outputs["receipt.json"], receipt)
    print(f"COMPLETE: {len(users)} users, {receipt['query_videos']} cached videos, {n_rows} query rows, "
          f"{context['frames']} distinct context frames; {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
