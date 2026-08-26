"""Read-only progress summary for the resumable feature exporter."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    args = parser.parse_args()
    if args.total <= 0:
        raise ValueError("total must be positive")
    uri = f"file:{args.database.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    completed = int(connection.execute("SELECT COUNT(*) FROM features").fetchone()[0])
    failures = 0
    by_partition = dict(connection.execute(
        "SELECT split || ':' || role, COUNT(*) FROM features GROUP BY split, role"
    ))
    connection.close()
    activity_paths = [args.database, Path(str(args.database) + "-wal")]
    modified_timestamp = max(path.stat().st_mtime for path in activity_paths if path.exists())
    modified = datetime.fromtimestamp(modified_timestamp, tz=timezone.utc).isoformat()
    print(json.dumps({
        "completed": completed,
        "total": args.total,
        "percent": round(100 * completed / args.total, 2),
        "last_database_activity_utc": modified,
        "failure_count": failures,
        "counts": by_partition,
        "eta": "unknown_without_mutating_runtime_rate",
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
