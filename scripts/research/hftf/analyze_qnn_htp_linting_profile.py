"""Compatibility shim for the staged DepthART diagnostics migration."""

from scripts.research.hftf.diagnostics.depthart.analyze_qnn_htp_linting_profile import *

if __name__ == "__main__":
    from scripts.research.hftf.diagnostics.depthart.analyze_qnn_htp_linting_profile import main
    main()
