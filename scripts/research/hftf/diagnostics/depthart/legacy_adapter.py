"""Staged compatibility imports for DepthART QNN/HTP diagnostics."""

import sys
from importlib import import_module
from pathlib import Path

_legacy_root = Path(__file__).resolve().parents[2]
if str(_legacy_root) not in sys.path:
    sys.path.insert(0, str(_legacy_root))

analyze_qnn_detailed_operator_profile = import_module(
    "scripts.research.hftf.analyze_qnn_detailed_operator_profile"
)
analyze_qnn_htp_linting_profile = import_module(
    "scripts.research.hftf.analyze_qnn_htp_linting_profile"
)
