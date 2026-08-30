# Codex workflow

1. Read `AGENTS.md`, `docs/PROJECT_STATE.md`, and the single affected route.
2. Check `git status --short`; when several lines are active, run
   `pwsh -NoProfile -File scripts/show_worktree_scope.ps1`; preserve unrelated work.
3. Make one scoped implementation pass.
4. Run the narrowest check that can falsify the change.
5. Update `docs/CURRENT_DECISION.md` only when status, claim, stop condition, or
   next action changed.
6. Stage only task-owned paths, make one focused commit, and push normally.
7. Release task-owned processes, worktrees, ports, and temporary resources.

Use `tools/ba.ps1` for workstation profiles and
`scripts/run_android_gradle.ps1` for Android. Historical research is read from
the archive tag only when explicitly required.
