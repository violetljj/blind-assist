"""Compatibility shim for the staged DepthART deployment migration."""

from scripts.research.hftf.deployment.depthart.depthart_admission_r1 import *

if __name__ == "__main__":
    from scripts.research.hftf.deployment.depthart.depthart_admission_r1 import main
    main()
