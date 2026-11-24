"""Render the two measured alert-cost frontiers from the retained receipts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]/"artifacts.local/work/egogeom-omni-v0/math_theory"


def main():
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.2),constrained_layout=True,sharey=True)
    for version,ax in zip(("v4","v41"),axes):
        fname="v5_empirical_report.json" if version=="v4" else "v5_empirical_report_v41.json"
        m6=json.loads((ROOT/"m6_evt_eprocess_risk"/fname).read_text())
        m8=json.loads((ROOT/HERE.name/fname).read_text())
        for data,held,color,label in [(m6["exact_geometry_frontier"],m6["geometry"]["pooled"],"#087e8b","Geometry"),
                                      (m8["curve"],m8["crossfit"]["pooled"],"#c45121","Stopping policy")]:
            # The post-hoc envelope and the learned fold-specific thresholds
            # have different policy families, so a marker may exceed a curve.
            points=data["pareto"]
            ax.step([r[3]/data["walker_km"] for r in points],[r[1] for r in points],
                    where="post",color=color,label=label+" (post hoc)")
            ax.scatter(held["clear_warnings_per_km"],held["timely"],facecolor=color,edgecolor="black",
                       s=44,zorder=3,label=label+" (held out)")
        ax.axvline(5,color="0.35",ls="--",lw=1.)
        ax.set(title="Frozen "+("v4" if version=="v4" else "v4.1"),xlim=(-.4,20),
               ylim=(-8,365),xlabel="Repeated clear warnings per walker-km")
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Timely alerts among 365 avoidable cases")
    axes[1].legend(fontsize=8,loc="upper left")
    fig.savefig(ROOT/HERE.name/"alert_tradeoff.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    main()
