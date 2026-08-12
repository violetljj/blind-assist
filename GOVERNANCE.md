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

Sustained contributors may be invited to maintain a bounded area after they
have demonstrated reliable review, respect for evidence and privacy rules, and
the ability to keep CI and documentation current. Access is expanded gradually
and may be withdrawn if it is unused or puts users, contributors, or repository
integrity at risk.

## Project continuity

If active maintenance pauses, the repository should say so explicitly. The
latest release, open issues, known evidence gaps, and security contact state
must not be presented as actively supported when they are not.
