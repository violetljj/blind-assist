# Security Policy

BlindAssist processes camera input and produces assistive feedback, so privacy, data handling, permissions, model integrity and device-integration flaws deserve private handling.

The current system, supply-chain, release, local-network, and model-assisted
maintenance boundaries are documented in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## Supported versions

Security fixes target the default `master` branch and the most recent GitHub release. Historical tags and experimental research applications may not receive backports.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting page:

<https://github.com/violetljj/blind-assist/security/advisories/new>

Do not open a public issue for an unpatched vulnerability. Include only the minimum information needed to reproduce it:

- affected commit, tag, module and Android version;
- impact and realistic attack or privacy scenario;
- reproduction steps or a minimal proof of concept;
- whether permissions, network access, camera data, logs, models or external devices are involved;
- any suggested mitigation.

Do not upload real bystander footage, private user data, credentials, signing material, device identifiers or restricted dataset samples. Use synthetic or redacted evidence whenever possible.

## Response and disclosure

The maintainer will triage reports on a best-effort basis, confirm the affected surface, and coordinate a fix and disclosure plan when the report is valid. Please allow a reasonable remediation period before public disclosure. This project does not offer a bug bounty or guaranteed response-time SLA.

## Safety boundary

A security fix does not certify BlindAssist as a safety device. Reports about model quality, accessibility, reliability or unsafe wording are still valuable, but they may be tracked separately from software vulnerabilities after sensitive details are removed.
