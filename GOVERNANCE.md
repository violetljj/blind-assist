# BlindAssist Governance

BlindAssist is a maintainer-led public project. The repository is currently
maintained by [@violetljj](https://github.com/violetljj). Contributions in
Chinese or English are welcome.

## How decisions are made

- Small, reversible fixes are decided through pull-request review and the
  applicable automated checks.
- Broad changes to architecture, permissions, packaged models, feedback
  behavior, evidence contracts, or licensing should start with a public issue.
- Decisions that change a stable interface or project-wide rule are recorded in
  the owning current document or an architecture decision record and linked
  from the pull request.
- Research results never receive default-App, deployment, or safety authority
  merely because their code is merged. Promotion uses the separate gates in
  [docs/RESEARCH_GOVERNANCE.md](docs/RESEARCH_GOVERNANCE.md).

The maintainer makes the final merge and release decision. Rejections should be
explained in terms of scope, evidence, maintenance cost, privacy, licensing, or
project safety boundaries rather than contributor identity.

## Transparency and conflicts

Project work should normally happen in public issues and pull requests. Private
channels are reserved for vulnerabilities, credentials, private data, and
other sensitive reports covered by [SECURITY.md](SECURITY.md).

Reviewers and maintainers should disclose a material conflict of interest. If
the only maintainer is conflicted, the limitation and the available evidence
should be recorded publicly without exposing sensitive information.

## Becoming a maintainer

Roles recognize demonstrated work; they are not rewards for stars, affiliation,
or private relationships. Time and pull-request counts are minimum signals, not
automatic promotion.

| Role | Minimum evidence | Repository authority |
| --- | --- | --- |
| Contributor | First qualifying pull request merged | No special access |
| Regular contributor | At least 3 merged pull requests across at least 4 weeks; responds to review and keeps scope verifiable | No special access; may be invited to own a follow-up issue |
| Triager / reviewer | Repeatedly reproduces issues, classifies scope, or reviews changes accurately while respecting privacy and evidence boundaries | Bounded issue-triage or review responsibility; no release control |
| Area maintainer | At least 2–3 months of sustained work, at least 5 high-quality merged pull requests, review of other contributors, and demonstrated understanding of the area's safety and evidence boundaries | Limited ownership through scoped `CODEOWNERS` or equivalent review responsibility |
| Core maintainer | Sustained cross-module decisions plus participation in releases and security-incident handling | Release, security, and cross-module authority granted explicitly |

The current maintainers evaluate quality, continuity, judgment, conflicts of
interest, and the project's actual need before each invitation. A candidate may
remain at a role without write access indefinitely; contribution quality is not
measured only by code volume.

Access is expanded gradually, scoped to the smallest necessary area, reviewed
periodically, and may be withdrawn if it is unused or puts users, contributors,
or repository integrity at risk. Maintainers with repository access must use an
individual account with two-factor authentication, never share credentials, and
follow protected-branch, pull-request, required-check, and no-force-push rules.
Until a second trusted maintainer exists, a pull-request requirement may use zero
mandatory approvals to avoid deadlocking the sole maintainer; required CI and the
public review record still apply. The approval count should become at least one
when an independent trusted reviewer can reliably satisfy it.

After a contributor's first qualifying pull request merges, the maintainer should
thank them in the relevant release notes and offer one bounded follow-up issue in
the same area. This is an invitation to continue, not a promise of promotion.

## Project continuity

If active maintenance pauses, the repository should say so explicitly. The
latest release, open issues, known evidence gaps, and security contact state
must not be presented as actively supported when they are not.
