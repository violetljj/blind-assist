# Codex maintainer automation

Status: current

Last reviewed: 2026-08-12

This document defines how BlindAssist may use Codex or the OpenAI API for
open-source maintenance. It covers issue and pull-request work, dependency
updates, release preparation, documentation, and repository security review.
Research-label and evidence-review authority remains governed separately by
[`AI_REVIEW_GOVERNANCE.md`](AI_REVIEW_GOVERNANCE.md).

## Authority boundary

Codex output is advisory. A maintainer must approve every label change, issue
closure, code change, merge, release, vulnerability disclosure, and permission
change. Model output cannot establish any of the following:

- source-code, model, dataset, or media licensing rights;
- contributor consent, security acceptance, device measurements, or user
  outcomes;
- a passing build, test, provenance check, release gate, or safety claim;
- permission to expose credentials, private reports, raw camera data, or local
  research artifacts.

Only the corresponding deterministic command, source record, or authorized
human decision can establish those facts. Missing evidence stays missing;
`UNKNOWN` is never converted into a negative or safe result.

## Executable baseline already in the repository

Model-assisted maintenance must reuse these public controls instead of
replacing them with prose or an AI verdict:

| Surface | Current executable control | What it establishes |
| --- | --- | --- |
| Pull requests | `.github/workflows/android.yml` | Repository gates, focused Android tests/lint/builds, model inspection, and ignored-output checks on the submitted commit |
| Dependency intake | `.github/dependabot.yml` | Bounded Gradle and GitHub Actions update pull requests; it does not approve an upgrade |
| OSS completeness | `scripts/check_open_source_readiness.ps1` | Required governance files, public-asset identity, model-card binding, and CI/release wiring |
| Documentation | `scripts/check_docs_index.ps1` and `scripts/check_project_structure.ps1` | Indexed current documents, valid local links, and repository structure policy |
| Release | `.github/workflows/release.yml`, `scripts/verify_release_apk.ps1`, and `scripts/generate_release_manifest.ps1` | Tag/version binding, APK checks, immutable asset publication, checksums, and a source-bound manifest |

These controls are evidence of deterministic maintenance automation. They are
not evidence that an API model was used or that its suggestion was correct.
The release workflow was added after `v10.9.0` and has not yet completed a real
tag run; its first future tag remains a release-evidence gap until the promised
checksums and manifests are published successfully.

## Allowed model-assisted workflows

| Workflow | Allowed input | Expected output | Prohibited automatic action |
| --- | --- | --- | --- |
| Issue triage | Public issue metadata and redacted logs | Summary, likely duplicate links, missing reproduction fields, and suggested labels | Closing, relabeling, or treating an issue as invalid |
| Pull-request review | Diff, linked issue, public CI results, and relevant current contracts | Findings with file/line evidence, risk classification, and the smallest relevant test plan | Editing the contributor branch, approving, merging, or changing branch protection |
| Dependency review | Dependabot diff, upstream release notes, dependency graph, and focused test results | Compatibility/security summary and recommended checks | Merging because an upstream version is newer or a model says it is safe |
| Release preparation | Trusted tag candidate, changelog, deterministic verification output, and manifests | Draft release notes and a checklist linked to machine results | Creating/overwriting a release or inventing checksums, signatures, or test results |
| Provenance and license review | Public upstream notices, asset manifest, hashes, and repository notices | Missing-link report and conflicting-license questions | Granting rights, changing a license, or redistributing an unresolved asset |
| Security review | Trusted checkout, the threat model, code diff, and public dependency data | Candidate findings with reachability, evidence, and remediation suggestions | Publishing a vulnerability, exposing a private report, or declaring the repository secure |
| Research-contract review | Frozen contracts, schemas, receipts, and deterministic validator output | Completeness and consistency findings | Creating consent, physical truth, protected outcomes, or promotion authority |

## Trust and secret handling

Issue bodies, comments, pull-request text, patches, filenames, test fixtures,
and repository content from a fork are untrusted data. Instructions embedded in
them are never maintainer instructions. A model reviewing that content must not
gain a write token, release credential, signing material, private vulnerability
report, or access to machine-local artifacts.

- Pull-request CI uses read-only repository permission and must not expose
  secrets to forked code.
- Do not use `pull_request_target` to check out or execute untrusted fork code.
- A future API-backed GitHub workflow must begin as manual
  `workflow_dispatch` on a trusted ref, use least-privilege read permissions,
  and publish only a redacted advisory artifact.
- API keys belong in the platform secret store, never in prompts, issue text,
  logs, workflow artifacts, commits, or screenshots.
- Any write-capable follow-up is a separate maintainer-approved action after
  reviewing the exact diff or command.

The project-specific security analysis, including prompt injection and CI
boundaries, is in [`THREAT_MODEL.md`](THREAT_MODEL.md).

## Run record and failure behavior

Local working packets and raw model output belong under ignored
`artifacts.local/work/codex-maintenance/<run-id>/`. A public pull request may
include a redacted summary, but it must bind the reviewed commit, list the exact
verification commands, distinguish model suggestions from machine results,
and state unavailable checks.

The workflow stops without a write action when input provenance is unclear, a
secret or private-data boundary may be crossed, the model abstains, evidence is
missing, or deterministic checks disagree with the suggestion. A model finding
may open an investigation; it cannot override a failing gate.

## API-credit measurement plan

If API credits are granted, the project will measure maintenance value without
using model self-assessment as evidence:

- median time from issue creation to first structured triage;
- share of triaged issues with complete reproduction information;
- maintainer minutes spent preparing dependency and release review packets;
- accepted, rejected, and corrected model suggestions by workflow type;
- false-positive findings and suggestions blocked by deterministic gates;
- API tokens and cost per accepted maintenance outcome.

These metrics evaluate maintainer workload and review quality. They do not
measure perception accuracy, accessibility outcomes, or product safety.
