"""Label moving LaMAR corridors against the independent CAB NavVis mesh."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from plyfile import PlyData
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

from arkit_selective_heading_dev_20261116 import (
    HALF_WIDTH, MIN_SOLID_ANGLE, angle_deg, in_corridor, samples,
)
from scenefun_faro_corridor_screen_dev_20261116 import category_for, summarize


MAX_RANGE = 8.0
SIDE_MARGIN = 0.05
ANCHOR_SPACING = 0.39
MAX_REFERENCE_GAP = 2.5


def world_from_scan(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("pose_graph_optimized, __absolute__,"):
            pose = np.array(line.split(",")[2:9], dtype=float)
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(pose[[1, 2, 3, 0]]).as_matrix()
            transform[:3, 3] = pose[4:]
            return transform
    raise ValueError(f"Missing published world transform in {path}")


def nearest(times, values):
    after = np.clip(np.searchsorted(times, values), 1, len(times) - 1)
    before = after - 1
    return np.where(np.abs(times[before] - values) <= np.abs(times[after] - values), before, after)


def walk_windows(data):
    times, world, local = (data[key] for key in ("ts", "T_wc", "T_local"))
    component = data.get("component", np.zeros(len(times), dtype=np.int32))
    gap = data["ref_gap_s"]
    aligned = np.isfinite(world).all(axis=(1, 2)) & (gap <= MAX_REFERENCE_GAP)
    depth_index = data["depth_index"]
    changes = dict(depth_frames=len(depth_index), depth_pose_matched=int((depth_index >= 0).sum()),
                   depth_aligned=int(aligned[depth_index[depth_index >= 0]].sum()), walk_windows=0,
                   alignment_jump_rejected=0)
    rows = []
    last = -np.inf
    for depth_row, i in enumerate(depth_index):
        if i < 0 or not aligned[i]:
            continue
        expected = times[i] + np.array([-.4, .4, .8])
        before, after, future = nearest(times, expected)
        window = slice(before, future + 1)
        if np.max(np.abs(times[[before, after, future]] - expected)) > .075 or not aligned[window].all():
            continue
        if (component[window] != component[i]).any() or np.max(np.diff(times[window])) > .22:
            continue
        delta = world[future, :3, 3] - world[i, :3, 3]
        run_time = times[future] - times[i]
        distance = np.linalg.norm(delta[:2])
        horizontal_head = world[i, :2, 2]
        if not .30 <= distance / run_time <= 2.3 or abs(delta[2]) / run_time > .25:
            continue
        if np.linalg.norm(horizontal_head) < .5:
            continue
        head = horizontal_head / np.linalg.norm(horizontal_head)
        travel = delta[:2] / distance
        yaw = angle_deg(head, travel)
        if yaw > 70.:
            continue
        world_local = world[i] @ np.linalg.inv(local[i])
        expected_world = world_local @ local[[before, after, future]]
        delta_world = expected_world[:, :3, 3] - world[[before, after, future], :3, 3]
        yaw_change = Rotation.from_matrix(
            np.swapaxes(expected_world[:, :3, :3], 1, 2) @ world[[before, after, future], :3, :3]
        ).magnitude().max()
        if np.max(np.linalg.norm(delta_world, axis=1)) > .10 or np.rad2deg(yaw_change) > 4.:
            changes["alignment_jump_rejected"] += 1
            continue
        changes["walk_windows"] += 1
        if times[i] - last >= ANCHOR_SPACING:
            rows.append((depth_row, i, future, [before, i, after], head, travel, yaw))
            last = times[i]
    return rows, changes


def keep_triangle_cells(cells, grid):
    """Retain triangles whose cell bounding boxes meet the crop, including long triangles."""
    lo, hi = cells.min(axis=1), cells.max(axis=1)
    shape = np.array(grid.shape)
    inside = ((lo < shape) & (hi >= 0)).all(axis=1)
    rows = np.flatnonzero(inside)
    a = np.maximum(lo[rows], 0)
    b = np.minimum(hi[rows], shape - 1) + 1
    prefix = np.pad(grid, [(1, 0)] * 3).cumsum(0).cumsum(1).cumsum(2)
    x0, y0, z0 = a.T
    x1, y1, z1 = b.T
    occupied = (prefix[x1, y1, z1] - prefix[x0, y1, z1] - prefix[x1, y0, z1] - prefix[x1, y1, z0]
                + prefix[x0, y0, z1] + prefix[x0, y1, z0] + prefix[x1, y0, z0] - prefix[x0, y0, z0])
    inside[rows] = occupied > 0
    return inside


def crop_mesh(path, centres, transform):
    """Mmap the large raw PLY and copy only faces usable by range-limited rays."""
    start_time = perf_counter()
    data = PlyData.read(path, mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    vertices = data["vertex"].data
    faces = data["face"].data["vertex_indices"]
    scan_centres = (centres - transform[:3, 3]) @ transform[:3, :3]
    cell_size = MAX_RANGE + .05
    origin = scan_centres.min(axis=0) - cell_size
    indices = np.floor((scan_centres - origin) / cell_size).astype(int)
    mask = np.zeros(tuple(indices.max(axis=0) + 2), dtype=bool)
    mask[tuple(indices.T)] = True
    mask = binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool))
    print(f"Cropping {len(faces):,} NavVis triangles to {mask.sum()} occupied ray tiles", flush=True)
    batch = 750_000
    cells = np.empty((len(vertices), 3), dtype=np.int16)
    for begin in range(0, len(vertices), batch):
        end = min(begin + batch, len(vertices))
        for axis, field in enumerate("xyz"):
            coord = np.floor((vertices[field][begin:end] - origin[axis]) / cell_size)
            cells[begin:end, axis] = np.clip(coord, -1, mask.shape[axis])
    selections = []
    for begin in range(0, len(faces), batch):
        part = faces[begin:begin + batch]
        keep = keep_triangle_cells(cells[part], mask)
        selections.append(part[keep])
    del cells
    chosen = np.concatenate(selections, axis=0)
    print(f"Selected {len(chosen):,} NavVis triangles", flush=True)
    del selections
    used = np.zeros(len(vertices), dtype=bool)
    used[chosen] = True
    xyz = np.column_stack([vertices[key][used] for key in "xyz"])
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    remap = np.cumsum(used, dtype=np.int32) - 1
    indices = remap[chosen]
    print(f"Cropped {len(xyz):,} vertices in {perf_counter() - start_time:.1f}s", flush=True)
    return xyz.astype(np.float32), indices


class NavVis:
    def __init__(self, vertices, faces):
        from embreex.mesh_construction import TriangleMesh
        from embreex.rtcore_scene import EmbreeScene

        self.scene = EmbreeScene()
        self.mesh = TriangleMesh(self.scene, vertices, faces)

    def distance(self, origin, rays):
        origin = np.broadcast_to(origin, rays.shape).astype(np.float32)
        return self.scene.run(origin, np.ascontiguousarray(rays, np.float32), query="DISTANCE")


def surface_rays():
    az, el = np.meshgrid(np.arange(-93, 94), np.arange(-75, 51), indexing="xy")
    az, el = np.deg2rad(az.ravel()), np.deg2rad(el.ravel())
    rays = np.column_stack([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    return rays, np.deg2rad(1.) ** 2 * np.cos(el)


def surface_views(scan, centres, forward, rays, weights):
    x, y = forward
    transform = np.array([[x, -y, 0.], [y, x, 0.], [0., 0., 1.]])
    rays = rays @ transform.T
    views = []
    for centre in centres:
        depth = scan.distance(centre, rays)
        visible = (depth > .08) & (depth < MAX_RANGE)
        views.append((centre + rays[visible] * depth[visible, None], weights[visible]))
    return views


def free_coverage(scan, target, centres):
    grid_shape = target.shape[:-1]
    target = target.reshape(-1, 3)
    seen = np.zeros(len(target), dtype=bool)
    for centre in centres:
        rays = target - centre
        ranges = np.linalg.norm(rays, axis=1)
        depth = scan.distance(centre, rays / np.maximum(.001, ranges[:, None]))
        seen |= (ranges > .10) & (depth > ranges + .08) & (depth < MAX_RANGE)
    return seen.reshape(grid_shape).mean(axis=1)


def evaluate_box(scan, origin, direction, centres, views, width):
    votes = np.array([in_corridor(view, origin, direction, width)[0] for view in views])
    coverage = free_coverage(scan, samples(origin, direction, width), centres)
    status = np.full(2, -1, dtype=np.int8)
    status[(votes >= MIN_SOLID_ANGLE).sum(axis=0) >= 2] = 1
    status[(votes == 0).all(axis=0) & (coverage >= .75)] = 0
    return status, coverage


def robust_box(scan, origin, direction, centres, views):
    state, cover = evaluate_box(scan, origin, direction, centres, views, HALF_WIDTH)
    inner, _ = evaluate_box(scan, origin, direction, centres, views, HALF_WIDTH - SIDE_MARGIN)
    outer, _ = evaluate_box(scan, origin, direction, centres, views, HALF_WIDTH + SIDE_MARGIN)
    robust = np.where(((state == 1) & (inner == 1)) | ((state == 0) & (outer == 0)), state, -1)
    return state.tolist(), robust.tolist(), cover.round(3).tolist()


def label_capture(scan, name, data, windows, stats):
    rays, weights = surface_rays()
    poses = data["T_wc"]
    rows = []
    for depth_row, i, future, at, head, travel, yaw in windows:
        origin, centres = poses[i, :3, 3], poses[at, :3, 3]
        view_axis = (head + travel) / np.linalg.norm(head + travel)
        views = surface_views(scan, centres, view_axis, rays, weights)
        # The reference includes a closer aligned view from the same walk.
        a, b, head_cover = robust_box(scan, origin, head, centres, views)
        c, d, travel_cover = robust_box(scan, origin, travel, centres, views)
        stamp = str(data["timestamp_us"][i])
        rows.append(dict(timestamp=stamp, time=float(data["ts"][i]), depth_row=depth_row,
                         direction_deg=round(yaw, 3), head=a, travel=c, head_robust=b, travel_robust=d,
                         head_coverage=head_cover, travel_coverage=travel_cover,
                         head_origin=origin.round(4).tolist(), future_stamp=int(data["timestamp_us"][future]),
                         category=category_for(a, c), robust_category=category_for(b, d)))
    return dict(visit=name, rows=rows, sampled=len(rows), **stats)


def self_check():
    cells = np.array([[[0, 0, 0], [2, 0, 0], [2, 1, 0]], [[0, 2, 0], [2, 2, 0], [1, 2, 0]]])
    occupied = np.zeros((3, 3, 3), dtype=bool)
    occupied[1, 0, 0] = True
    np.testing.assert_array_equal(keep_triangle_cells(cells, occupied), [True, False])
    plane = np.array([[0., -6., -6.], [0., 6., -6.], [0., -6., 6.], [0., 6., 6.]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    origin = np.array([0., 0., 1.6])
    forward = np.array([1., 0.])
    for wall_x, expected in ((1.1, [1, 1]), (3.0, [0, 0])):
        plane[:, 0] = wall_x
        scan = NavVis(plane, faces)
        centres = np.array([origin + [-.2, 0., 0.], origin, origin + [.2, 0., 0.]])
        views = surface_views(scan, centres, forward, *surface_rays())
        assert robust_box(scan, origin, forward, centres, views)[0] == expected
    assert not np.any(free_coverage(scan, samples(origin, -forward, HALF_WIDTH), centres))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("navvis", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("trajectory", nargs="+", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    self_check()
    inputs = []
    centres = []
    for path in args.trajectory:
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in ("ts", "timestamp_us", "T_wc", "T_local", "component",
                                                  "ref_gap_s", "depth_index")}
        windows, stats = walk_windows(data)
        print(path.stem, stats, len(windows), flush=True)
        inputs.append((path.stem, data, windows, stats))
        centres.extend(data["T_wc"][at, :3, 3] for _, _, _, at, *_ in windows)
    if not centres:
        raise ValueError("No independently aligned walk windows")
    transform = world_from_scan(args.navvis / "alignment_global.txt")
    xyz, faces = crop_mesh(args.navvis / "mesh_simplified.ply", np.concatenate(centres), transform)
    scan = NavVis(xyz, faces)
    with (args.output / "cropped_mesh.npz").open("xb") as stream:
        np.savez(stream, vertices=xyz, faces=faces)
    print("Running independent near-field raycasting", flush=True)
    visits = []
    for name, data, windows, stats in inputs:
        visit = label_capture(scan, name, data, windows, stats)
        visits.append(visit)
        print(name, {k: v["episodes"] for k, v in summarize([visit]).items()}, flush=True)
    nominal = summarize(visits)
    robust = summarize([dict(visit, rows=[dict(r, category=r["robust_category"]) for r in visit["rows"]])
                        for visit in visits])
    result = dict(visits=visits, nominal=nominal, robust=robust, wall_seconds=round(perf_counter() - start, 3))
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    for key, label in [("nominal", nominal), ("width_robust", robust)]:
        print(key, {k: (v["frames"], v["episodes"], v["visits"]) for k, v in label.items()})


if __name__ == "__main__":
    main()
