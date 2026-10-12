"""Frozen six-visit band hybrid acquisition; no model inference or score reads."""
from pathlib import Path
import argparse
from rgb_near_readout_scale_acquire import acquire

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--wall-s", type=float, default=220)
    a = p.parse_args()
    acquire(a.root, a.wall_s, target_count=6, download_limit=1_000_000_000,
            cohort_prefix="bandhybrid_arkit_", split_order=("Validation", "Training"))
