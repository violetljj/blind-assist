#!/usr/bin/env python3
"""Render one public ProcTHOR-10K house and export topology/runtime facts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.util.lock import Lock
from ai2thor.wsgi_server import WsgiServer
from PIL import Image


DATASET = Path("/data/test.jsonl.gz")
OUTPUT = Path("/output")


def main() -> int:
    # Docker Desktop's Linux-volume fcntl state can remain spuriously blocked
    # after a stopped first-download container.  This task runs exactly one
    # controller writer, so use AI2-THOR's Windows-equivalent no-op lock policy.
    Lock.lock = lambda self: None
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        house = json.loads(next(stream))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    controller = Controller(
        scene=house,
        platform=CloudRendering,
        width=320,
        height=320,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        server_class=WsgiServer,
    )
    try:
        reachable_event = controller.step(action="GetReachablePositions")
        reachable = reachable_event.metadata.get("actionReturn") or []
        Image.fromarray(reachable_event.frame).save(OUTPUT / "frame.png")
        doors = [
            {
                "id": door.get("id"),
                "room0": door.get("room0"),
                "room1": door.get("room1"),
                "openable": door.get("openable"),
                "openness": door.get("openness"),
            }
            for door in house.get("doors", [])
        ]
        receipt = {
            "schema_version": "blindassist_procthor_runtime_smoke_v1",
            "last_action_success": reachable_event.metadata.get("lastActionSuccess"),
            "reachable_position_count": len(reachable),
            "door_count": len(doors),
            "doors": doors,
            "agent": reachable_event.metadata.get("agent"),
            "frame_path": str((OUTPUT / "frame.png").resolve()),
        }
        (OUTPUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
