from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.research.dual_loop_causal_track_tristate_r0.cli import main


if __name__ == "__main__":
    main()
