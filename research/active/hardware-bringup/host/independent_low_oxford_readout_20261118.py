"""Read the linked Oxford object scores, inspect matched negative pixels and plot both approaches."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[4] / "artifacts.local/work/independent-low-20261118"


def load(path):
    return json.loads(path.read_bytes())


def replay(manifest):
    key = hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
    folder = ROOT / "replay" / key
    rows = load(folder / "predictions.json")["rows"]
    return folder, {(r["seq"], r["i"]): r for r in rows}


def support_area(path, select):
    with np.load(path) as data:
        height, width = data["shape"]
        if select.shape != (height, width):
            raise ValueError("Reference mask is outside the calibrated prediction grid")
        idx = data["pred_indices"]
        masks = np.unpackbits(data["pred_masks"], axis=-1, count=int(width), bitorder="little")
        support = masks.reshape(3, 3, -1)[:, :, idx]
        return (support * select.ravel()[idx] * data["pred_area_m2"]).sum(-1)


def negative_checks():
    manifest = ROOT / "local/negative_generic_ground.json"
    data = load(manifest)
    base = Path(data["data_root"])
    folder, predictions = replay(manifest)
    substrate = []
    for row in data["frames"]:
        pred = predictions[row["seq"], row["i"]]
        if pred["evidence"]["sem80"]["LOW"] or not pred["evidence"]["geom"]["LOW"]:
            continue
        z = cv2.imread(str(base / row["depth"]), cv2.IMREAD_UNCHANGED) * row["ref_scale"]
        k = np.asarray(row["K"])
        normal, offset = row["ground"]
        v, u = np.indices(z.shape)
        height = offset + z * (normal[0]*(u-k[0, 2])/k[0, 0] + normal[1]*(v-k[1, 2])/k[1, 1] + normal[2])
        area = lambda mask: float(support_area(folder / pred["support_path"], mask)[2, 0])
        substrate.append(dict(seq=row["seq"], i=row["i"], t=row["t"],
            total_m2=area(np.ones_like(z, dtype=bool)), reference_ground_m2=area((z>.3) & (abs(height)<=.10))))
    (ROOT / "local/negative_matched_pixels.json").write_text(json.dumps(substrate, indent=1), encoding="utf-8")
    return substrate


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    cases = load(ROOT / "scoring/oxford-review-locked.json")["cases"]
    folder, predictions = replay(ROOT / "local/filled_generic_ground.json")
    fig = plt.figure(figsize=(9.8, 6.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=.06, right=.94, top=.80, bottom=.145, hspace=.48, wspace=.24)
    fig.text(.06,.95,"同一次接近，能否多保住一个报警？",fontsize=21,weight="bold",va="top")
    fig.text(.06,.88,"牛津两处设施 · 黄色轮廓为独立激光参考 · 右图仅计物体自身的预测低障面积",fontsize=11)
    colors = ["#69528b", "#007665"]
    labels = ["仅阈值 0.80", "几何保护 0.80"]
    for row, case in enumerate(cases):
        observations = [o for o in case["observations"] if o["i"] in case["matched_indices"]]
        last = observations[-1]
        photo = cv2.imread(last["rgb"])
        outline = cv2.imread(last["reference_footprint"], 0)
        contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(photo, contours, -1, (35,222,249), 3)
        pic, chart = fig.add_subplot(grid[row,0]), fig.add_subplot(grid[row,1])
        h = photo.shape[0]
        pic.imshow(cv2.cvtColor(photo[round(.47*h):round(.88*h)], cv2.COLOR_BGR2RGB))
        pic.axis("off")
        pic.set_title(["广场单级石缘","建筑入口多级楼梯"][row], loc="left", fontsize=13, weight="bold")
        distances, values = [], []
        for obs in observations:
            pred = predictions[obs["seq"], obs["i"]]
            roi = cv2.imread(obs["reference_footprint"], 0) > 0
            values.append(support_area(folder / pred["support_path"], roi)[1:,0] * 10000)
            distances.append(obs["near"]["forward_m"])
        x = np.arange(len(values))
        for i, values_by_arm in enumerate(np.asarray(values).T):
            chart.plot(x, values_by_arm, ("o-","s--")[i], lw=(2.8,1.7)[i], color=colors[i], ms=(7,4)[i])
        chart.axhline(50, color="#8c6849", ls=":", lw=1.4)
        chart.text(.06, 57, "报警门槛 50", color="#765233", fontsize=8)
        chart.set(ylim=(-15,411), xlim=(-.14,len(x)-.86), yticks=[0,100,200,300,400])
        chart.set_xticks(x,[f"{v:.2f} m" for v in distances])
        chart.set_title("低障报警面积（cm²）", loc="left", fontsize=11)
        chart.grid(axis="y", color="#e3e8e7", lw=.7)
        chart.spines[["top","right"]].set_visible(False)
        chart.set_xlabel("按接近时刻排列的参考距离", fontsize=9)
    handles = [Line2D([0],[0], color=c, marker=m, ls=s, label=label) for c,m,s,label in
               zip(colors, ("o","s"), ("-","--"), labels)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.95,.87), ncol=2, frameon=False, fontsize=9)
    fig.text(.06,.055,"图像来源  Oxford Spires Dataset，Tao 等，CC BY-NC-SA 4.0 · 已裁剪并标注。",fontsize=8)
    fig.savefig(ROOT / "scoring/oxford-paired-near-events.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for frame in negative_checks():
        print(frame["i"], round(frame["total_m2"],4), round(frame["reference_ground_m2"],4))
    figure()
