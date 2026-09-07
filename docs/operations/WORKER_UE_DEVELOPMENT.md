# Worker UE development

Use the existing UU Remote installation for interactive editing on the worker.
SSH and the scheduled job runner handle compilation, scripted scene expansion,
capture and postprocessing. UU sign-in and an interactive remote connection are
separate from the recorded command-line acceptance checks.

## Editing and source ownership

The worker desktop shortcut **BlindAssist UE Development** opens an independent
Content/Config copy of the laboratory. Its path is stored in worker-local
`config/environment.json` as `ueDevelopmentProject`; entering the environment
exposes `BLINDASSIST_UE_DEVELOPMENT_PROJECT`. The copy initially contains 11,388
content files (16,248,412,011 bytes). It shares the existing DDC through
`ueDdcPath`, but does not share writable Content with the capture project.

The installed `scripts/Open-UE-Editor.ps1` comes from
[`worker_ue_editor.ps1`](../../tools/worker_ue_editor.ps1). Run it from the worker's
interactive desktop with `-WorkerRoot <workspace>`; `-ShowCommand` resolves and
checks paths without opening an editor. Keep the editor closed when another
task needs its RAM/VRAM. Closing a task-owned editor must not stop another user
or job's editor session.

New scenes should use versioned map/asset names. Before acquisition, publish only
the required assets and scripts into a task-owned capture input, record hashes,
and run a small scoped check. Never overwrite frozen maps or infer permission to
rerun a consumed cohort from a successful environment check. Keep bulk assets,
DDC and raw captures on the worker; return code, manifests and selected previews
to the controller for integration. Interactive work can happen on either machine;
the controller remains the source integration authority.

## Build entrypoints

The worker uses UE 5.8.2 installed-engine development files, Visual Studio 2022
Build Tools 17.14, MSVC 14.44, Windows SDK 10.0.26100.0 and .NET Framework 4.8 SDK.
Resolve physical paths from `UE_ENGINE_ROOT`, `BLINDASSIST_MSVC_ROOT` and the
worker-local configuration. The shared Visual Studio root is deliberately short
because the installer rejects long installation roots. Its shared/cache paths
use Windows backslashes. Windows SDK registration uses the installer-managed
system location; project outputs remain on the artifact volume.
The physical engine root also uses a short path within the worker artifact
tree; the previous toolchain path remains a compatibility junction. Read the
current configured root rather than hard-coding a former installation path.
Worker-user UBT configuration directs subsequent Unreal Build Accelerator storage
to `artifacts/work/cache/uba`; the first canary's launch used the original system
default. The detached runner preserves `UE-LocalDataCachePath` for DDC reuse.

For a new native capture plugin package, use a fresh output:

```powershell
python tools/build_ue_capture_plugin.py --engine $env:UE_ENGINE_ROOT --output "$env:BLINDASSIST_ARTIFACTS/evidence/UNIQUE_PLUGIN_BUILD"
if ($LASTEXITCODE -ne 0) { throw 'Plugin build failed; inspect its retained receipt' }
```

For a C++ project, invoke the installed engine's `Engine/Build/BatchFiles/Build.bat`
with `<ProjectName>Editor Win64 Development -Project=<absolute.uproject> -WaitMutex`.
Use the [worker runner](WORKER_HANDOFF.md) for detached builds. Compiling a new
plugin does not replace `BLINDASSIST_UE_CAPTURE_PLUGIN`: keep the accepted capture
binary pinned until the new combination has its own acceptance check.

This is Windows project/plugin development using an installed engine. Engine
source rebuilding, other platform SDKs and arbitrary third-party plugins have
their own dependencies. Provisioning is not proof that every UE target builds.

## Provisioning evidence

Worker-local `artifacts/evidence/ue-development-20260908` retains installer
results, the additive engine manifest, extraction receipts, independent project
copy receipt, launcher configuration and C++ acceptance outputs. The payload has
165,408 files plus supplemental inputs: engine/plugin source, intermediate build inputs and static
libraries. Existing engine files are hash-checked and never overwritten.
Archive hashes and per-file hashes bind the transfer to the matching Build.version.
Failed installation/extraction attempts remain diagnostic evidence. A separate
43-record supplement supplies UAT's `Engine/Intermediate/ScriptModules` index.
Another 174-file supplement restores platform extension rules/source and related
build inputs. The union covers all 2,418 `Build.cs`/`Target.cs` files found in the
source engine audit; this inventory coverage is distinct from build acceptance.
The correction restores real source modules named `DerivedDataCache` and `Logs`;
directory basenames alone cannot safely distinguish source from runtime caches.

The C++ canary compiled in 152.24 seconds, including first-build rule/PCH work.
Its editor loaded the native function (returned 42) and saved a `.umap` and
material `.uasset` using NullRHI. The initial rendering attempt triggered cold
SM5 shader compilation and was stopped; asset/API validation does not require
rendering. A postprocessing path error was corrected by resolving UE-relative
asset paths, without rerunning the successful editor asset operation. Receipts
retain these attempts and process release. This timing is a first-build sample,
not an incremental-build or sustained-throughput benchmark.
The native capture plugin also completed UAT BuildPlugin packaging (about
182 seconds end-to-end), with UBA storage verified on the worker artifact volume.
Its DLL is retained as build evidence; the existing accepted capture plugin stays
pinned. No new capture cohort or rendered-image equivalence check was run.

Use the retained terminal receipts, not the existence of a folder or a dispatch
acknowledgement, to assess completed capability. Follow
[batch capture](UE_CAPTURE_BATCH.md) for acquisition throughput and resume.
