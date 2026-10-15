"""One-shot, CPU-only worker status and completed-UMD byte mirror.

No resident monitor, training, GPU job, process control, or scientific evaluation.
Run repeatedly from an existing scheduler; successful mirrors are idempotent.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REMOTE = "G:/DevWorkspace/BlindAssist/artifacts/work/functional-parts-full-20261015"
os.environ.setdefault("OMP_NUM_THREADS", "1")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def contained_file(path, root):
    root = Path(root).resolve()
    candidate = Path(path).resolve(strict=True)
    if not candidate.is_file() or not candidate.is_relative_to(root):
        raise ValueError("Referenced file must resolve inside the declared artifact root")
    return candidate, candidate.relative_to(root).as_posix()


def valid_remote_root(value):
    normal = str(PureWindowsPath(value)).replace("\\", "/").rstrip("/")
    if not re.fullmatch(r"[A-Za-z]:/[A-Za-z0-9_./ -]+", normal) or ".." in PurePosixPath(normal).parts:
        raise ValueError("Remote root must be a simple absolute Windows artifact path")
    if "/artifacts/work/" not in normal.lower():
        raise ValueError("Remote root must be below the worker canonical artifacts/work tree")
    return normal


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def sanitize(value, config):
    sensitive = [str(config.get(name, "")) for name in ("identityFile", "host", "user")]
    if isinstance(value, str):
        for secret in sensitive:
            if len(secret) > 2:
                for variant in {secret, secret.replace("\\", "/"), secret.replace("/", "\\")}:
                    value = value.replace(variant, "[worker-config-redacted]")
        return value
    if isinstance(value, list):
        return [sanitize(item, config) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item, config) for key, item in value.items()
                if not re.search(r"password|secret|credential|identityfile|privatekey", key, re.I)}
    return value


class Worker:
    def __init__(self, config_path):
        self.config = load(config_path)
        c = self.config
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", c["host"]):
            raise ValueError("Invalid configured SSH endpoint")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", c["user"]):
            raise ValueError("Invalid configured SSH user")
        if not re.fullmatch(r"[A-Za-z]:[\\/][A-Za-z0-9_./\\-]+", c["remotePowerShell"]):
            raise ValueError("Configured remote PowerShell must be a simple executable path")
        if not Path(c["identityFile"]).is_file():
            raise FileNotFoundError("Configured SSH identity is unavailable")
        self.endpoint = c["user"] + "@" + c["host"]
        self.options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
                        "-o", "IdentitiesOnly=yes", "-i", c["identityFile"]]

    def command(self, script, timeout=60):
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        command = self.config["remotePowerShell"] + " -NoProfile -OutputFormat Text -EncodedCommand " + encoded
        if len(command) > 7600:
            raise ValueError("Encoded PowerShell exceeds the bounded Windows SSH command size")
        try:
            result = subprocess.run(["ssh", *self.options, self.endpoint, command], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("SSH timed out; remote state remains unknown, no jobs were redispatched") from None
        if result.returncode:
            raise RuntimeError("SSH failed: " + sanitize(result.stderr[-1600:], self.config))
        # Environment entrypoints may print a banner; our final result has a marker.
        lines = [line for line in result.stdout.splitlines() if line.startswith("SYNC_JSON:")]
        if not lines:
            raise RuntimeError("Remote command did not return its bounded JSON result")
        return sanitize(json.loads(lines[-1][10:]), self.config)

    def upload(self, local, remote, timeout=1800):
        try:
            result = subprocess.run(["scp", "-q", *self.options, str(local), self.endpoint + ":" + remote],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("SCP timed out; task-owned partial payload retained for inspection") from None
        if result.returncode:
            raise RuntimeError("SCP failed; task-owned partial payload retained: " + sanitize(result.stderr[-1200:], self.config))


def status_script(remote_root):
    # Shared read handles avoid holding an exclusive reader lock on active JSON.
    # Errors from a concurrent atomic replace/partial JSON remain explicit.
    return r"""$ErrorActionPreference='Stop';$r=ROOT;
function ReadMeta($p,$tail=0){try{$f=[IO.File]::Open($p,'Open','Read',([IO.FileShare]::ReadWrite-bor[IO.FileShare]::Delete));try{if($tail){[void]$f.Seek([math]::Max(0,$f.Length-$tail),'Begin')};$s=[IO.StreamReader]::new($f);try{$t=$s.ReadToEnd()}finally{$s.Dispose()};if($tail){$t}else{$t|ConvertFrom-Json -Depth 35}}finally{$f.Dispose()}}catch{@{read_error=$_.Exception.GetType().Name}}};
$a=@();$e=@();foreach($d in @('queue','post_queue','runs/segformer','runs/dino','plsp/train_seed10000','plsp/final_seen_test','post/wait_main','post/wait_umd','transfer/segformer/export','transfer/dino/export','migration','data/umd','final')){$p=Join-Path $r $d;if(Test-Path -LiteralPath $p){$fs=@(Get-ChildItem -LiteralPath $p -File -ErrorAction SilentlyContinue);foreach($f in $fs){if($f.Name -in @('terminal.json','queue_terminal.json','queue_owner.json','queue.lock.json','owner.lock','progress.json','progress.jsonl','plsp_asset_receipt.json','bootstrap_receipt.json')){$v=if($f.Name -eq 'progress.jsonl'){ReadMeta $f.FullName 4096}else{if($f.Length -lt 1048576){ReadMeta $f.FullName}else{@{oversize=$true}}};$a+=@{path=$f.FullName.Substring($r.Length+1);modified_utc=$f.LastWriteTimeUtc.ToString('o');bytes=$f.Length;value=$v}}};$e+=@($fs|Where-Object{$_.Name -match 'stderr.*\.log$|error.*\.log$'}|Sort-Object LastWriteTimeUtc -Descending|Select-Object -First 3|ForEach-Object{@{path=$_.FullName.Substring($r.Length+1);tail=ReadMeta $_.FullName 4096}})}};
$o=Get-CimInstance Win32_OperatingSystem;$ps=@(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'"|Where-Object{$_.CommandLine -like ('*'+$r.Replace('/','\')+'*') -or $_.CommandLine -like ('*'+$r+'*')}|Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine);$g=@(& nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>$null);$m=Join-Path $r 'migration/umd_mirror_receipt.json';$mirror=if(Test-Path -LiteralPath $m){ReadMeta $m}else{$null};$j=@{status='READ_ONLY_SNAPSHOT';utc=[DateTime]::UtcNow.ToString('o');root=$r;files=$a;last_errors=$e;processes=$ps;ram=@{free_mib=[math]::Round($o.FreePhysicalMemory/1024);total_mib=[math]::Round($o.TotalVisibleMemorySize/1024)};gpu=$g;gpu_query_exit=$LASTEXITCODE;umd_mirror=$mirror};'SYNC_JSON:'+($j|ConvertTo-Json -Depth 45 -Compress)
""".replace("ROOT", ps_quote(remote_root))


def build_package(local_root, destination):
    """Read metadata, stream raw file bytes into ZIP_STORED; never decode pixels."""
    local_root = Path(local_root).resolve()
    umd = local_root / "data/umd"
    supervisor = umd / "supervisor_terminal.json"
    if load(supervisor).get("status") != "COMPLETE":
        raise ValueError("Local UMD supervisor is not COMPLETE")
    records_path = umd / "shared_label_records.jsonl"
    frozen_bytes = records_path.read_bytes()
    records = [json.loads(line) for line in frozen_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    if not records:
        raise ValueError("Completed UMD records are empty")
    files, kinds = {}, {}
    for record in records:
        if record.get("manual_annotation_status") != "AUTHOR_SELECTED_MANUAL":
            raise ValueError("UMD records contain a non-author-selected manual frame")
        for field in ("image", "mask"):
            file, rel = contained_file(record[field], local_root)
            files[rel], kinds[rel] = file, "author_manual_" + field
        provenance = record.get("manual_index_provenance")
        if provenance:
            file, rel = contained_file(provenance, local_root)
            files[rel], kinds[rel] = file, "metadata_evidence"
    # Include complete metadata evidence without live downloader ownership/state.
    required = ["shared_label_records.jsonl", "shared_label_records_receipt.json", "manual_index_census.json",
                "manual_index_provenance.json", "manual_frame_manifest.jsonl", "terminal.json", "supervisor_terminal.json"]
    optional = ["census.json", "frame_manifest.jsonl", "native_text_files.json", "download_receipt.json",
                "author_code_index.json", "code_request_reservations.json", "http_probe.json"]
    for name in required + optional:
        file = umd / name
        if name in optional and not file.exists():
            continue
        file, rel = contained_file(file, local_root)
        files[rel], kinds[rel] = file, "metadata_evidence"
    # Author text referenced by the evidence is small and remains raw bytes.
    text_index = umd / "native_text_files.json"
    if text_index.exists():
        for item in load(text_index):
            file, rel = contained_file(item["path"], local_root)
            files[rel], kinds[rel] = file, "author_native_text"
    manifest = {"status": "PACKAGED", "format": "UMD_RAW_BYTE_MIRROR_V1", "records": len(records),
                "records_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
                "supervisor_terminal_sha256": sha(supervisor), "compression": "ZIP_STORED", "workers": 1,
                "mask_pixels_decoded": False, "scientific_scores_computed": False,
                "supervisor_relative_path": supervisor.relative_to(local_root).as_posix(), "files": []}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for rel in sorted(files):
            file = files[rel]
            before = file.stat()
            checksum = hashlib.sha256()
            with file.open("rb") as source, archive.open("payload/" + rel, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    checksum.update(block)
                    target.write(block)
            after = file.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("Source changed during byte packaging; partial archive retained")
            manifest["files"].append({"path": rel, "bytes": before.st_size,
                                       "sha256": checksum.hexdigest(), "kind": kinds[rel]})
        archive.writestr("mirror_manifest.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))
    if records_path.read_bytes() != frozen_bytes or sha(supervisor) != manifest["supervisor_terminal_sha256"]:
        raise RuntimeError("Frozen metadata changed while packaging")
    os.replace(partial, destination)
    manifest["package_sha256"] = sha(destination)
    manifest["package_bytes"] = destination.stat().st_size
    save(destination.with_suffix(".manifest.json"), manifest)
    return manifest


# Standard-library-only receiver. All paths are checked before any extraction.
# It verifies the entire ZIP before opening it, verifies every member checksum,
# preserves raw record bytes, and publishes supervisor success last.
INSTALLER = r'''import hashlib,json,os,pathlib,sys,time,zipfile
root=pathlib.Path(sys.argv[1]).resolve(); package=pathlib.Path(sys.argv[2]).resolve(); expected=sys.argv[3]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def inside(path):
 p=path.resolve()
 if not p.is_relative_to(root):raise ValueError('Mirror destination escapes artifact root')
 return p
inside(package)
if sha(package)!=expected:raise ValueError('Uploaded package SHA mismatch')
receipt=root/'migration/umd_mirror_receipt.json'
with zipfile.ZipFile(package) as z:
 m=json.loads(z.read('mirror_manifest.json')); sup=m['supervisor_relative_path']; seen=set()
 if m['format']!='UMD_RAW_BYTE_MIRROR_V1':raise ValueError('Unknown mirror format')
 for row in m['files']:
  rel=pathlib.PurePosixPath(row['path'])
  if rel.is_absolute() or '..' in rel.parts or ':' in row['path'] or '\\' in row['path'] or row['path'] in seen:raise ValueError('Invalid mirror path')
  seen.add(row['path']); inside(root/row['path'])
 if set(z.namelist())!={'mirror_manifest.json'}|{'payload/'+r for r in seen}:raise ValueError('Unexpected/duplicate ZIP members')
 if len(z.namelist())!=len(seen)+1:raise ValueError('Duplicate ZIP names')
 if sup not in seen:raise ValueError('Missing final supervisor receipt')
 existing=[];copied=[]
 for row in sorted(m['files'],key=lambda r:(r['path']==sup,r['path'])):
  target=inside(root/row['path']); target.parent.mkdir(parents=True,exist_ok=True)
  if row['path']==sup:
   if sha(root/'data/umd/shared_label_records.jsonl')!=m['records_sha256']:raise ValueError('Frozen records byte checksum mismatch before supervisor publication')
  if target.exists():
   if target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:raise ValueError('Divergent existing mirror file: '+row['path'])
   existing.append(row['path']);continue
  temp=inside(target.with_name(target.name+'.sync-'+expected[:12]+'.partial')); h=hashlib.sha256();size=0
  with z.open('payload/'+row['path']) as source,temp.open('wb') as dest:
   for b in iter(lambda:source.read(1024*1024),b''):dest.write(b);h.update(b);size+=len(b)
  if size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Extracted member SHA mismatch')
  if row['path']==sup:
   if json.loads(temp.read_text(encoding='utf-8-sig')).get('status')!='COMPLETE':raise ValueError('Supervisor success not COMPLETE')
   if sha(temp)!=m['supervisor_terminal_sha256']:raise ValueError('Supervisor byte receipt mismatch')
  os.replace(temp,target);copied.append(row['path'])
 records=root/'data/umd/shared_label_records.jsonl'
 if sha(records)!=m['records_sha256']:raise ValueError('Frozen records byte checksum mismatch')
 result={'status':'COMPLETE','package_sha256':expected,'records_sha256':m['records_sha256'],'supervisor_terminal_sha256':m['supervisor_terminal_sha256'],'records':m['records'],'verified_files':len(seen),'copied_files':len(copied),'already_identical_files':len(existing),'mask_pixels_decoded':False,'scientific_scores_computed':False,'supervisor_published_last':True,'ended_unix':time.time()}
 temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');os.replace(temp,receipt)
 print('SYNC_JSON:'+json.dumps(result))
'''


def mirror_if_ready(local_root, remote_root, worker, snapshot):
    migration = local_root / "migration"
    supervisor = local_root / "data/umd/supervisor_terminal.json"
    if not supervisor.exists() or load(supervisor).get("status") != "COMPLETE":
        return {"status": "WAITING_LOCAL_UMD_COMPLETE", "transferred": False}
    record_sha = sha(local_root / "data/umd/shared_label_records.jsonl")
    supervisor_sha = sha(supervisor)
    remote = snapshot.get("umd_mirror") or {}
    if remote.get("status") == "COMPLETE" and remote.get("records_sha256") == record_sha and remote.get("supervisor_terminal_sha256") == supervisor_sha:
        result = {"status": "ALREADY_MIRRORED", "transferred": False, "remote_receipt": remote}
        save(migration / "umd_mirror_receipt.json", result)
        return result
    package = migration / "umd_author_manual_mirror.zip"
    saved_manifest = package.with_suffix(".manifest.json")
    manifest = load(saved_manifest) if saved_manifest.exists() and package.exists() else None
    if not manifest or manifest.get("records_sha256") != record_sha or manifest.get("supervisor_terminal_sha256") != supervisor_sha or sha(package) != manifest.get("package_sha256"):
        manifest = build_package(local_root, package)
    remote_dir = remote_root + "/migration"
    worker.command("$ErrorActionPreference='Stop';[void][IO.Directory]::CreateDirectory(" + ps_quote(remote_dir) + ");'SYNC_JSON:{}'")
    remote_package = remote_dir + "/umd_author_manual_" + manifest["package_sha256"][:16] + ".zip"
    receiver = migration / "install_umd_mirror.py"
    receiver.write_text(INSTALLER, encoding="utf-8")
    remote_receiver = remote_dir + "/install_umd_mirror_" + sha(receiver)[:16] + ".py"
    worker.upload(package, remote_package)
    worker.upload(receiver, remote_receiver, timeout=60)
    entry = ps_quote(worker.config["enterScript"])
    command = "$ErrorActionPreference='Stop';. " + entry + " -Profile 'research';& python " + " ".join(
        ps_quote(value) for value in (remote_receiver, remote_root, remote_package, manifest["package_sha256"]))
    command += ";if($LASTEXITCODE -ne 0){throw 'UMD mirror receiver failed'}"
    result = worker.command(command, timeout=1800)
    if result.get("status") != "COMPLETE" or result.get("records_sha256") != record_sha:
        raise RuntimeError("Remote mirror did not return matching COMPLETE evidence")
    save(migration / "umd_mirror_receipt.json", {"status": "COMPLETE", "transferred": True,
                                                "remote_receipt": result, "package_sha256": manifest["package_sha256"]})
    return {"status": "COMPLETE", "transferred": True, "remote_receipt": result}


def structured_snapshot(snapshot):
    for item in snapshot.get("files", []):
        if item.get("path", "").endswith("progress.jsonl") and isinstance(item.get("value"), str):
            text = item["value"]
            lines = [line for line in text.splitlines() if line.strip()]
            parsed = None
            skipped = 0
            for line in reversed(lines):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    skipped += 1
            item["value"] = {"last_complete_record": parsed, "trailing_partial_lines_skipped": skipped,
                             "bounded_tail_bytes": 4096}
    return snapshot


def acquire_sync_lock(path):
    """OS-released, nonblocking local lock; retained file is not a daemon/lease."""
    stream = Path(path).open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        return None
    return stream


def synthetic_check(root):
    fixture = root / "tmp" / ("worker_sync_metadata_check_" + uuid.uuid4().hex[:10])
    umd = fixture / "data/umd"
    umd.mkdir(parents=True)
    (umd / "fake-image.bin").write_bytes(b"synthetic image bytes; no decoder")
    (umd / "fake-mask.mat").write_bytes(b"synthetic mask bytes; no decoder")
    for name in ("shared_label_records_receipt.json", "manual_index_census.json", "manual_index_provenance.json", "terminal.json"):
        save(umd / name, {"synthetic": True})
    save(umd / "supervisor_terminal.json", {"status": "COMPLETE", "synthetic": True})
    (umd / "manual_frame_manifest.jsonl").write_bytes(b'{"synthetic":true}\n')
    frozen = (json.dumps({"image": str(umd / "fake-image.bin"), "mask": str(umd / "fake-mask.mat"),
                          "manual_annotation_status": "AUTHOR_SELECTED_MANUAL"}) + "\n").encode()
    (umd / "shared_label_records.jsonl").write_bytes(frozen)
    package = fixture / "migration/synthetic.zip"
    manifest = build_package(fixture, package)
    with zipfile.ZipFile(package) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert archive.read("payload/data/umd/shared_label_records.jsonl") == frozen
        for row in manifest["files"]:
            content = archive.read("payload/" + row["path"])
            assert hashlib.sha256(content).hexdigest() == row["sha256"] and len(content) == row["bytes"]
    blocked = False
    try:
        contained_file(Path(__file__), fixture)
    except ValueError:
        blocked = True
    assert blocked
    # Execute the actual standard-library receiver on generated bytes only.
    receiver = fixture / "receiver.py"
    receiver.write_text(INSTALLER, encoding="utf-8")
    target = fixture / "receiver_artifact"
    target.mkdir()
    remote_package = target / "migration/payload.zip"
    remote_package.parent.mkdir()
    shutil.copyfile(package, remote_package)
    result = subprocess.run([sys.executable, str(receiver), str(target), str(remote_package), manifest["package_sha256"]],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    receipt = load(target / "migration/umd_mirror_receipt.json")
    assert receipt["status"] == "COMPLETE" and receipt["supervisor_published_last"]
    assert (target / "data/umd/shared_label_records.jsonl").read_bytes() == frozen
    repeated = subprocess.run([sys.executable, str(receiver), str(target), str(remote_package), manifest["package_sha256"]],
                              capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 0, repeated.stderr
    repeated_receipt = load(target / "migration/umd_mirror_receipt.json")
    assert repeated_receipt["copied_files"] == 0
    invalid_target = fixture / "invalid_records_receiver"
    invalid_target.mkdir()
    invalid_package = invalid_target / "invalid.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(invalid_package, "w", compression=zipfile.ZIP_STORED) as dest:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename == "mirror_manifest.json":
                invalid_manifest = json.loads(content)
                invalid_manifest["records_sha256"] = "0" * 64
                content = json.dumps(invalid_manifest).encode()
            dest.writestr(info.filename, content)
    rejected = subprocess.run([sys.executable, str(receiver), str(invalid_target), str(invalid_package), sha(invalid_package)],
                              capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    assert not (invalid_target / "data/umd/supervisor_terminal.json").exists()
    assert not (invalid_target / "migration/umd_mirror_receipt.json").exists()
    first_lock = acquire_sync_lock(fixture / "test_sync.lock")
    assert first_lock is not None and acquire_sync_lock(fixture / "test_sync.lock") is None
    first_lock.close()
    recovered_lock = acquire_sync_lock(fixture / "test_sync.lock")
    assert recovered_lock is not None
    recovered_lock.close()
    sample_snapshot = structured_snapshot({"files": [{"path": "runs/test/progress.jsonl",
        "value": 'partial leading bytes\n{"updates":12}\n{"updates":'}]})
    assert sample_snapshot["files"][0]["value"]["last_complete_record"] == {"updates": 12}
    assert sample_snapshot["files"][0]["value"]["trailing_partial_lines_skipped"] == 1
    value = {"status": "PASS", "checks": ["ZIP_STORED_single_worker", "all_member_SHA256", "frozen_records_bytes",
             "outside_root_rejected", "receiver_SHA_and_extract", "supervisor_success_last", "idempotent_receiver",
             "partial_JSONL_snapshot_last_complete_record", "invalid_records_blocks_supervisor_success",
             "OS_lock_prevents_overlap_and_releases"],
             "synthetic_only": True, "GPU_started": False, "fixture": str(fixture)}
    save(root / "migration/worker_sync_synthetic_check.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE)
    parser.add_argument("--config", type=Path, default=ROOT / "artifacts.local/work/remote-worker/worker.json")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    local = args.local_root.resolve(strict=True)
    artifact_root = (ROOT / "artifacts.local").resolve(strict=True)
    if not local.is_relative_to(artifact_root):
        raise ValueError("Local root must resolve inside canonical project artifacts")
    # Packaging is single-threaded and yields ordinary below-normal CPU priority.
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel.GetPriorityClass.restype = wintypes.DWORD
        handle = kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(handle, 0x4000) or kernel.GetPriorityClass(handle) != 0x4000:
            raise ctypes.WinError(ctypes.get_last_error())
    if args.self_test:
        result = synthetic_check(local)
        print(json.dumps({"status": result["status"], "synthetic_checks": len(result["checks"])}))
        return
    remote_root = valid_remote_root(args.remote_root)
    worker = Worker(args.config)
    destination = local / "migration/remote_status.json"
    state = {"started_unix": time.time(), "status_only": args.status_only, "GPU_jobs_started": 0,
             "remote_root": remote_root, "script_sha256": sha(__file__)}
    try:
        snapshot = structured_snapshot(worker.command(status_script(remote_root)))
        state.update(status="SUCCESS", snapshot=snapshot)
        save(destination, state)
        if not args.status_only:
            guard = acquire_sync_lock(local / "migration/worker_sync.lock")
            if guard is None:
                state["UMD_mirror"] = {"status": "SYNC_ALREADY_ACTIVE", "transferred": False}
            else:
                try:
                    state["UMD_mirror"] = mirror_if_ready(local, remote_root, worker, snapshot)
                finally:
                    guard.close()
    except Exception as error:
        state.update(status="FAILED", error=sanitize(type(error).__name__ + ": " + str(error), worker.config))
        raise RuntimeError(state["error"]) from None
    finally:
        state["wall_seconds"] = time.monotonic() - started
        save(destination, state)
    print(json.dumps({"status": state["status"], "status_only": args.status_only,
                      "UMD_mirror": state.get("UMD_mirror", {}).get("status"), "wall_seconds": state["wall_seconds"]}))


if __name__ == "__main__":
    main()
