# Codex Skills Snapshot Manifest

- Snapshot date: `2026-05-22 00:22 +08:00`
- Source on old computer: `C:\Users\junjie\.codex\skills`
- Archive: `codex/skills-snapshot/codex-skills-20260522.zip`
- Archive size: `14,005,316 bytes`
- SHA256: `A64F7287A463BDB12BFB16D57F1538D77D3D71B95DBC088EC2F0BC6A4ABB4F1C`
- Archive entries verified with `tar -tf`: `13,446`
- Source files counted before packaging: `13,031`
- Source size before packaging: `20.47 MB`
- Restore script verification: extracted to a temporary repository-local
  `.codex` folder and verified `13,031` restored files across `72` top-level
  skill directories.

The archive contains the full `skills/` directory snapshot, including the
Codex-managed `.system` folder and the existing `.name-cn-backup-20260516`
backup folder. On a new computer, restoring the snapshot is enough to recreate
the local skill files used by this project history.

## Restore

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\restore_codex_skills.ps1
```

Then restart Codex so it reloads the restored skills.

## Important Notes

- The snapshot is for development continuity. Do not treat it as an Android app
  dependency.
- A filename scan found no private cookie database or `.env` secret file in the
  skills snapshot, but the archive still contains local development tooling and
  should be handled like project-maintainer material.
- If Codex on the new computer already ships a newer `.system` skill set, you
  may choose to restore only user-installed skill folders manually from the
  archive instead of overwriting `.system`.
