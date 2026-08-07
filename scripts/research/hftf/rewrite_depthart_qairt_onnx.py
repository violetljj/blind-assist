"""Compatibility shim for the staged DepthART deployment migration."""

from scripts.research.hftf.deployment.depthart.rewrite_depthart_qairt_onnx import *

if __name__ == "__main__":
    from scripts.research.hftf.deployment.depthart.rewrite_depthart_qairt_onnx import main
    main()
