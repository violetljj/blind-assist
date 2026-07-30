from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import acquire_labels, evaluate, freeze_confirmation, produce


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--labels-inventory", type=Path, required=True)
    freeze.add_argument("--timestamps-zip", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--freeze", type=Path, required=True)
    acquire.add_argument(
        "--role", choices=("source_2d", "truth_3d"), required=True
    )
    acquire.add_argument("--output-root", type=Path, required=True)
    acquire.add_argument("--receipt", type=Path, required=True)

    source = subparsers.add_parser("produce")
    source.add_argument("--freeze", type=Path, required=True)
    source.add_argument("--source-root", type=Path, required=True)
    source.add_argument("--source-acquisition-receipt", type=Path, required=True)
    source.add_argument("--truth-root", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--receipt", type=Path, required=True)

    validation = subparsers.add_parser("evaluate")
    validation.add_argument("--freeze", type=Path, required=True)
    validation.add_argument("--truth-root", type=Path, required=True)
    validation.add_argument(
        "--truth-acquisition-receipt", type=Path, required=True
    )
    validation.add_argument("--producer-output", type=Path, required=True)
    validation.add_argument("--producer-receipt", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_confirmation.run(
            args.labels_inventory, args.timestamps_zip, args.output
        )
    elif args.command == "acquire":
        result = acquire_labels.run(
            args.freeze, args.role, args.output_root, args.receipt
        )
    elif args.command == "produce":
        result = produce.run(
            args.freeze,
            args.source_root,
            args.source_acquisition_receipt,
            args.truth_root,
            args.output,
            args.receipt,
        )
    elif args.command == "evaluate":
        result = evaluate.run(
            args.freeze,
            args.truth_root,
            args.truth_acquisition_receipt,
            args.producer_output,
            args.producer_receipt,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
