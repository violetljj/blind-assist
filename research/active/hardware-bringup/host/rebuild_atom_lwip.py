"""Build isolated AtomS3R qio_opi lwIP; never modify the installed Arduino SDK.

Requires the pinned M5Stack 3.3.8 SDK and esp-x32 2601 toolchain. Outputs must
reside below this checkout's artifacts.local. Link the returned liblwip.a ahead
of the SDK's -llwip; changing sketch defines alone cannot change prebuilt lwIP.
"""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import subprocess
import tarfile
import urllib.request


REPO = Path(__file__).resolve().parents[4]
IDF = "735507283d"
LWIP = "fd432e4ee2cfb7f7f1c7eb7227e0173412e7b84e"
LWIP_ARCHIVE_SHA256 = "99edee6321fc4e77d279a22313013085ca0c4d90b82ca6f1f79bde6d8aae7780"
PROBE_KEYS = ("TCP_SND_BUF", "TCP_SND_QUEUELEN", "TCP_SNDLOWAT",
              "TCP_SNDQUEUELOWAT", "sizeof_tcp_pcb", "offsetof_snd_buf", "sizeof_snd_buf")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    require(result.returncode == 0, result.stdout + result.stderr)
    return result.stdout


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-lwip-rebuild"})
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(request, timeout=60).read()
                path.write_bytes(data)
                break
            except OSError:
                if attempt == 2:
                    raise
    return path


def symbols(nm, library):
    return sorted({line.split()[-1] for line in run(
        [nm, "-g", "--defined-only", library]).splitlines() if len(line.split()) == 3})


def response_file(path, args):
    # GCC response files avoid Windows' 32767-character command-line limit.
    path.write_text("\n".join('"' + str(x).replace("\\", "/").replace('"', '\\"') + '"'
                              for x in args), encoding="utf-8")
    return "@" + str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sdk-dir", type=Path, help="Default: local-config.json arduino_data / packages/m5stack/tools/esp32s3-libs/3.3.8")
    parser.add_argument("--send-buffer", default=32768, type=int)
    args = parser.parse_args()
    require(5744 <= args.send_buffer <= 65535, "Send buffer must stay within the existing 16-bit ABI (5744..65535).")
    root = args.output_dir.resolve()
    artifacts = (REPO / "artifacts.local").resolve()
    require(root != artifacts and root.is_relative_to(artifacts), "--output-dir must be a directory below this checkout's artifacts.local.")
    if args.sdk_dir is None:
        config = json.loads((artifacts / "hardware-bringup/local-config.json").read_text(encoding="utf-8-sig"))
        args.sdk_dir = Path(config["arduino_data"]) / "packages/m5stack/tools/esp32s3-libs/3.3.8"
    sdk = args.sdk_dir.resolve()
    tools = sdk.parent.parent / "esp-x32/2601/bin"
    exe = ".exe" if (tools / "xtensa-esp32s3-elf-gcc.exe").exists() else ""
    tool = lambda name: tools / ("xtensa-esp32s3-elf-" + name + exe)
    versions = (sdk / "versions.txt").read_text()
    require("esp-idf: v5.5.4 735507283d" in versions and "lib-builder: master 14a0af9" in versions,
            "This script supports only the verified M5Stack 3.3.8 / IDF735507283d SDK.")
    original = sdk / "lib/liblwip.a"
    sdk_files = [sdk / "versions.txt", original, sdk / "qio_opi/include/sdkconfig.h"]
    sdk_files += list((sdk / "flags").glob("*")) + sorted((sdk / "include/lwip").rglob("*.h"))
    inputs = {
        "schema": 1, "idf": IDF, "lwip": LWIP, "send_buffer": args.send_buffer,
        "sdk": str(sdk), "sdk_sha256": {str(p.relative_to(sdk)): sha(p) for p in sdk_files},
        "compiler_sha256": sha(tool("gcc")), "compiler_version": run([tool("gcc"), "--version"]),
        "recipe_sha256": sha(Path(__file__)), "lwip_archive_sha256": LWIP_ARCHIVE_SHA256,
    }
    library = root / "liblwip.a"
    manifest_path = root / "rebuild-manifest.json"
    if library.exists():
        require(manifest_path.exists(), "Existing library has no manifest; choose a fresh output directory.")
        manifest = json.loads(manifest_path.read_text())
        require(manifest["inputs"] == inputs, "Existing build parameters or SDK differ; choose a fresh output directory.")
        require(manifest["library_sha256"] == sha(library), "Existing library hash differs from manifest.")
        require(all(sha(root / p) == digest for p, digest in manifest["source_sha256"].items()), "Existing source provenance differs.")
        print(f"Reused verified library: {library}")
        return

    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    tree = json.loads(fetch(f"https://api.github.com/repos/espressif/esp-idf/git/trees/{IDF}?recursive=1",
                            root / "idf-tree.json").read_text())
    require(not tree.get("truncated"), "Incomplete IDF source tree.")
    entries = [x for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith("components/lwip/")]

    def download(entry):
        relative = entry["path"].removeprefix("components/lwip/")
        require(".." not in PurePosixPath(relative).parts, "Invalid source path.")
        path = fetch(f"https://raw.githubusercontent.com/espressif/esp-idf/{IDF}/{entry['path']}", source / relative)
        data = path.read_bytes()
        require(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["sha"],
                f"IDF Git blob hash mismatch: {relative}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, entries))
    archive = fetch(f"https://codeload.github.com/espressif/esp-lwip/tar.gz/{LWIP}", root / "esp-lwip.tar.gz")
    require(sha(archive) == LWIP_ARCHIVE_SHA256, "Pinned lwIP archive hash mismatch.")
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name).relative_to("esp-lwip-" + LWIP)
            require(".." not in relative.parts, "Invalid archive path.")
            path = source / "lwip" / str(relative)
            data = tar.extractfile(member).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                require(path.read_bytes() == data, f"Cached lwIP source mismatch: {relative}")
            else:
                path.write_bytes(data)

    members = run([tool("ar"), "t", original]).splitlines()
    sources = []
    for path in re.findall(r'"([^"\n]+\.c)"', (source / "CMakeLists.txt").read_text()):
        path = path.replace("${target}", "esp32xx")
        if path not in sources and Path(path).name + ".obj" in members:
            sources.append(path)
    require(sorted(Path(p).name + ".obj" for p in sources) == sorted(members), "Source list differs from SDK archive members.")
    flags = shlex.split((sdk / "flags/c_flags").read_text())
    flags += ["@" + str(sdk / "flags/defines"), "-Os", "-DESP_LWIP_COMPONENT_BUILD",
              "-Wno-address", "-Wno-type-limits", "-Wno-array-parameter"]
    flags += ["-I" + str(sdk / "include" / p) for p in re.findall(
        r"-iwithprefixbefore\s+(\S+)", (sdk / "flags/includes").read_text())]
    flags += ["-I" + str(source), "-I" + str(source / "port/esp32xx/include")]
    config = (sdk / "qio_opi/include/sdkconfig.h").read_text()
    require(re.search(r"#define CONFIG_LWIP_TCP_SND_BUF_DEFAULT 5744\b", config), "Unexpected baseline TCP send buffer.")
    probes = {}
    commands = []
    for label, size in [("baseline-probe", 5744), ("candidate", args.send_buffer)]:
        out = root / label
        out.mkdir(exist_ok=True)
        changed = re.sub(r"(#define CONFIG_LWIP_TCP_SND_BUF_DEFAULT )5744\b", rf"\g<1>{size}", config)
        (out / "sdkconfig.h").write_text(changed)
        common = [tool("gcc"), response_file(out / "compile-flags.rsp", ["-I" + str(out), *flags])]
        probe = out / "config_probe.c"
        probe.write_text('#include "lwip/tcp.h"\n#include <stddef.h>\nconst unsigned probe[]={TCP_SND_BUF,TCP_SND_QUEUELEN,TCP_SNDLOWAT,TCP_SNDQUEUELOWAT,sizeof(struct tcp_pcb),offsetof(struct tcp_pcb,snd_buf),sizeof(((struct tcp_pcb*)0)->snd_buf)};\n')
        run([*common, "-c", probe, "-o", out / "config_probe.o"])
        run([tool("objcopy"), "-O", "binary", "--only-section=.rodata.probe", out / "config_probe.o", out / "config_probe.bin"])
        probes[label] = dict(zip(PROBE_KEYS, struct.unpack("<7I", (out / "config_probe.bin").read_bytes())))
        require(probes[label]["TCP_SND_BUF"] == size, "Compiled TCP send buffer does not match request.")
        if label == "baseline-probe":
            continue
        require(all(probes[label][key] == probes["baseline-probe"][key] for key in PROBE_KEYS[4:]), "TCP PCB ABI changed.")

        def compile_one(path):
            obj = out / "objects" / (path + ".obj")
            obj.parent.mkdir(parents=True, exist_ok=True)
            command = [*common, "-c", source / path, "-o", obj]
            run(command)
            return obj, [str(x) for x in command]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            built = list(pool.map(compile_one, sources))
        commands = [x[1] for x in built]
        pending = root / "liblwip.pending.a"
        require(not pending.exists(), "Incomplete earlier archive exists; choose a fresh output directory.")
        run([tool("ar"), response_file(out / "archive.rsp", ["qc", pending, *[x[0] for x in built]])])
        run([tool("ar"), "s", pending])
        require(sorted(run([tool("ar"), "t", pending]).splitlines()) == sorted(members), "Built archive member mismatch.")
        exported = symbols(tool("nm"), pending)
        require(exported == symbols(tool("nm"), original), "Built archive public symbols differ from SDK.")
        pending.rename(library)
    manifest = {
        "inputs": inputs, "library_sha256": sha(library), "library": str(library),
        "members": len(members), "defined_symbols": len(exported), "probes": probes,
        "idf_git_blobs_verified": len(entries),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()},
    }
    (root / "commands.json").write_text(json.dumps(commands, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Built verified library: {library}\nSHA256: {manifest['library_sha256']}\nNo SDK changes or device operations performed.")


if __name__ == "__main__":
    main()
