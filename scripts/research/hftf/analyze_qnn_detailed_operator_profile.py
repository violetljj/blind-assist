"""Compatibility shim for the staged DepthART diagnostics migration."""

from scripts.research.hftf.diagnostics.depthart.analyze_qnn_detailed_operator_profile import *

if __name__ == "__main__":
    from scripts.research.hftf.diagnostics.depthart.analyze_qnn_detailed_operator_profile import main
    main()
