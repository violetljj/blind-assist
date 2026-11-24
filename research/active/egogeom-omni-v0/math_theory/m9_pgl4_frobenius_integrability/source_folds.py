"""Score fixed predictions on M3's pre-existing physical-source folds."""
import argparse
import hashlib
import re

import numpy as np
from eval_empirical import ART, OUTPUT, SUITES, fixed_threshold, sha, store, support_metrics, warnings


def main(version):
    results = {}
    for suite, (folder, _) in SUITES.items():
        def source(stem):
            return stem if suite.startswith("arkit") else re.fullmatch(r"(.+)_c\d+", stem).group(1)
        files = sorted((ART/"zeroshot_eval"/folder).rglob("*.npz"))
        names = sorted({source(p.stem) for p in files},key=lambda s:hashlib.sha256(("M3-alert-cost-v2/"+s).encode()).digest())
        folds = [[],[]]
        for path in files:
            pred = OUTPUT/version/suite/path.name
            with np.load(pred) as obs, np.load(path) as ref:
                folds[names.index(source(path.stem))%2].append({
                    "name":path.stem,"original_bev":obs["original_bev"],"joint_bev":obs["joint_bev"],
                    "eval_rows":ref["eval_rows"]})
        results[suite] = []
        for fold, clips in enumerate(folds):
            metrics = {}
            for arm in ("original_bev","joint_bev"):
                fixed = fixed_threshold(clips,arm)
                repeated = warnings(clips,arm)
                metrics[arm] = {"support":support_metrics(clips,arm),"fixed_2m":fixed,"repeated_clear":repeated,
                                "repeated_clear_per_km":repeated/fixed["walker_km"]}
            results[suite].append({"fold":fold,"physical_sources":names[fold::2],"metrics":metrics})
    out = OUTPUT/version
    store(out/"v5_source_folds.json",{"scores":results,"empirical_report_sha256":sha(out/"v5_empirical_report.json")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version",choices=("v4","v41"),default="v4")
    main(parser.parse_args().version)
