"""Compatibility shim for the staged DepthART deployment migration."""

from scripts.research.hftf.deployment.depthart.rewrite_depthart_qairt_hygiene import *

if __name__ == "__main__":
    from scripts.research.hftf.deployment.depthart.rewrite_depthart_qairt_hygiene import main
    main()
