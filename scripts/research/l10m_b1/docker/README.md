# Local Docker Codex worker

This image is only a model/search worker. It must not contain the B1 evaluator,
hidden cohort, truth, repository checkout, or Codex memories.

The controller mounts only a job-owned empty workspace and, when explicitly
authorized, the host ChatGPT login file read-only at `/root/.codex/auth.json`.
The container is run with a read-only root filesystem, dropped capabilities,
`no-new-privileges`, a bounded tmpfs, and no host path other than the worker
workspace and auth file. The image carries the pinned Google Trust Services
WE1 intermediate used by the local HTTPS proxy; the canary must be rerun if
the local proxy's presented issuer changes.

The Codex invocation uses `danger-full-access` only inside this hard container,
because the Docker Desktop kernel does not permit Codex's nested bubblewrap
namespace. The outer Docker boundary remains authoritative: the root filesystem
and worker workspace are read-only, capabilities are dropped, and no evaluator
or hidden data is mounted.

Build with:

```text
docker build --pull --build-arg CODEX_VERSION=0.148.0 -t l10m-b1-codex:0.148.0 scripts/research/l10m_b1/docker
```

Before a formal run, execute a canary that asks the worker to read a marker
outside `/workspace`; the command must fail and the worker must still return a
candidate. The evaluator remains local and authoritative.
