# Maintainer automation

The repository keeps a small set of deterministic local/CI gates:

| Surface | Command |
| --- | --- |
| Layout and active-route count | `scripts/check_project_structure.ps1` |
| Hot documentation links | `scripts/check_docs_index.ps1` |
| Generated files, secrets, pinned dependencies | `scripts/check_repo_hygiene.ps1` |
| Public maintenance and asset identity | `scripts/check_open_source_readiness.ps1` |
| Android build/test | `scripts/run_android_gradle.ps1 <tasks>` |

Run only the checks relevant to the changed surface. Release builds and public
asset changes additionally use `scripts/generate_release_manifest.ps1` and the
release workflow.
