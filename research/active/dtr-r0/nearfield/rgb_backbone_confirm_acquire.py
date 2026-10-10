"""Fixed six-visit, 1GB fresh confirmation reference acquisition; no inference."""
from __future__ import annotations
import argparse
from pathlib import Path
from rgb_near_readout_scale_acquire import acquire


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wall-s", type=float, default=325)
    args = parser.parse_args()
    acquire(args.root, args.wall_s, target_count=6, download_limit=1_000_000_000,
            cohort_prefix="confirm_arkit_")
