## What changed

Describe the focused change and link the issue it addresses.

## Why

Explain the user, contributor or maintainer impact.

## Verification

List exact commands and results. State any unavailable checks and the remaining evidence gap.

```text
command -> result
```

## Boundaries and risk

- Default App impact: `none` / explain
- Permissions, network, camera or private-data impact: `none` / explain
- Research/evidence role: `not applicable` / Development / other explicitly authorized role
- Safety claim: `none`

## Checklist

- [ ] The diff is scoped and contains no generated, private, credential or machine-local files.
- [ ] `git diff --check` passes.
- [ ] Relevant tests, lint, build and repository gates pass.
- [ ] Documentation and indexes are updated where stable responsibilities changed.
- [ ] `UNKNOWN`, failed evidence and historical terminals are preserved.
- [ ] Synthetic or model-reviewed evidence is labeled and not presented as user, deployment or safety proof.
