"""Шар перешкод за контактом з підлогою: предмети за семантичною розміткою (labels у pred/*.npz),
положення -- з точки, де предмет торкається підлоги (перетин променя з площиною підлоги за висотою
дрона), висота -- з верхнього краю сегмента. Не залежить від монокулярної глибини, тож тонкі
предмети (ніжки, триноги, стійки, рамки) потрапляють на карту навіть там, де глибина їх «не бачить».

  python contact_layer.py <лог АС Р1> --out <карта.npz> [--max-eps 0.012] [--no-contact]
Будує карту АС Р1 (build_map) і додає колонки контакту; пише <карта>_summary.json, як buildmap.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ac1_evacmap"))
from flight_log import FlightLog                              # noqa: E402
from mono_depth import floor_depth, recover_scale, flying_pixel_mask, backproject   # noqa: E402
from occupancy_map import OccupancyMap                        # noqa: E402
import app as ac1                                             # noqa: E402

# класи ADE20K, що стоять на підлозі (SegFormer-B0): меблі, коробки, сумки, стійки
OBSTACLE_IDS = {10: "cabinet", 15: "table", 17: "plant", 19: "chair", 23: "sofa", 24: "shelf", 30: "armchair",
                31: "seat", 33: "desk", 35: "wardrobe", 38: "railing", 41: "box", 45: "counter", 62: "bookcase",
                69: "bench", 73: "kitchen island", 77: "bar", 93: "pole", 97: "ottoman", 110: "stool",
                112: "basket", 115: "bag", 117: "cradle", 138: "ashcan", 146: "radiator"}
L_CONTACT = 2.0            # приріст логарифма шансів за одне спостереження контакту (l_occ АС Р1 = 0,85)


def contact_columns(cam, labels, floor_mask, height, yaw_deg, position, ids=OBSTACLE_IDS, stride=3,
                    min_range=0.3, max_range=3.5, max_height=2.2):
    """Список (x, y, z_top, клас) для стовпців зображення, де предмет стоїть на видимій підлозі."""
    H, W = labels.shape
    obstacle = np.isin(labels, list(ids))
    z_floor = floor_depth(cam, yaw_deg, height)
    d_c = cam.pixel_dirs()
    R = cam.rotation_world(yaw_deg)
    pos = np.asarray(position, float)
    out = []
    for u in range(0, W, stride):
        col_o, col_f = obstacle[:, u], floor_mask[:, u]
        v = H - 1
        while v > 0:
            if col_o[v] and v + 1 < H and col_f[v + 1:min(H, v + 4)].any():
                v_b = v
                v_t = v_b
                while v_t > 0 and col_o[v_t - 1]:
                    v_t -= 1
                zf = z_floor[v_b + 1, u]
                if np.isfinite(zf):
                    p_ground = pos + (R @ (d_c[v_b + 1, u] * zf))
                    D = float(np.hypot(*(p_ground[:2] - pos[:2])))
                    if min_range < D < max_range:
                        dir_t = R @ d_c[v_t, u]
                        horiz = float(np.hypot(dir_t[0], dir_t[1]))
                        z_top = pos[2] + dir_t[2] * (D / horiz) if horiz > 1e-6 else max_height
                        z_top = float(np.clip(z_top, 0.0, max_height))
                        if z_top > 0.15:
                            out.append((float(p_ground[0]), float(p_ground[1]), z_top, int(labels[v_b, u])))
                v = v_t - 1
            else:
                v -= 1
    return out


def add_contacts(occ, cols, l_contact=L_CONTACT):
    """Додає колонки контакту в карту логарифмів шансів; повертає кількість вокселів."""
    n = 0
    seen = set()
    for x, y, z_top, _cls in cols:
        i, j = int((x - occ.origin[0]) / occ.voxel), int((y - occ.origin[1]) / occ.voxel)
        if not (0 <= i < occ.shape[0] and 0 <= j < occ.shape[1]):
            continue
        k_top = min(occ.shape[2] - 1, int(z_top / occ.voxel))
        for k in range(0, k_top + 1):
            if (i, j, k) in seen:
                continue
            seen.add((i, j, k))
            occ.logodds[i, j, k] = min(occ.l_max, occ.logodds[i, j, k] + l_contact)
            n += 1
    return n


def build_map_metric(log, cam, max_eps=None, gamma=0.08, stride=2, max_range=4.0, seed=0):
    """Карта за метричною глибиною (pred: metric_depth, Depth Anything V2 Metric Indoor). Масштаб
    кадру додатково звіряється з підлогою: медіана z_floor / Z по пікселях підлоги (якщо підлоги
    >= 5 % кадру і розкид малий); без підлоги береться Z як є. Кадри без metric_depth пропускаються."""
    room = log.meta.get("room_size", [12.0, 12.0, 3.0])
    occ = OccupancyMap(size_xyz=tuple(room), voxel=0.1, max_range=max_range)
    rows = []
    for rec in log.records:
        if not log.has_pred(rec.index):
            continue
        z = np.load(log.pred_path(rec.index))
        if "metric_depth" not in z:
            continue
        Z = z["metric_depth"].astype(np.float32)
        mask = z["floor_mask"]
        zf = floor_depth(cam, rec.yaw_deg, rec.z_tof)
        sel = mask & np.isfinite(zf) & (zf > 0.3) & (zf < 8.0) & (Z > 0.2)
        scale, eps, accepted = 1.0, None, True
        if sel.sum() >= 0.05 * mask.size:
            ratio = zf[sel] / Z[sel]
            scale = float(np.median(ratio))
            eps = float(np.median(np.abs(ratio / scale - 1.0)))
            if max_eps is not None and eps > max_eps:
                accepted = False
        Zs = Z * scale
        Zs[(Zs < 0.2) | (Zs > 12.0)] = np.nan
        keep = flying_pixel_mask(Zs, gamma=gamma)
        sub = np.zeros_like(keep)
        sub[::stride, ::stride] = True
        n_pts = 0
        if accepted:
            pts = backproject(cam, Zs, rec.yaw_deg, (rec.x, rec.y, rec.z_tof), keep=keep & sub)
            n_pts = occ.integrate_points((rec.x, rec.y, rec.z_tof), pts, eps=eps if eps is not None else 0.05)
        rows.append({"index": rec.index, "waypoint": rec.waypoint, "yaw_deg": rec.yaw_deg, "z_tof": rec.z_tof,
                     "accepted": bool(accepted), "n_floor": int(mask.sum()), "n_inliers": int(sel.sum()),
                     "eps": eps, "alpha": scale, "beta": 0.0, "points": n_pts, "time_s": 0.0})
    return occ, rows


def build_map_plus(log_dir, out_npz, max_eps=None, contact=True, verbose=False, depth="rel"):
    log = FlightLog.load(log_dir)
    cam = ac1.camera_from_meta(log.meta)
    if depth == "metric":
        occ, rows = build_map_metric(log, cam, max_eps=max_eps)
    else:
        occ, rows = ac1.build_map(log, max_eps=max_eps, verbose=verbose)
    n_cols = n_vox = 0
    classes = {}
    if contact:
        for rec in log.records:
            if not log.has_pred(rec.index):
                continue
            z = np.load(log.pred_path(rec.index))
            if "labels" not in z:
                continue
            row = next((r for r in rows if r["index"] == rec.index), None)
            if row is None or not row["accepted"]:
                continue                        # лише кадри, де підлогу знайдено й масштаб прийнято
            cols = contact_columns(cam, z["labels"], z["floor_mask"], rec.z_tof, rec.yaw_deg,
                                   (rec.x, rec.y, rec.z_tof))
            n_cols += len(cols)
            n_vox += add_contacts(occ, cols)
            for _x, _y, _zt, c in cols:
                classes[OBSTACLE_IDS.get(c, str(c))] = classes.get(OBSTACLE_IDS.get(c, str(c)), 0) + 1
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    occ.save(out_npz, extra={"source": log.root})
    rep = ac1.write_report(rows, occ, os.path.splitext(out_npz)[0],
                           {"contact_columns": n_cols, "contact_voxels": n_vox, "contact_classes": classes,
                            "depth": depth})
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=float, default=None)
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--depth", choices=("rel", "metric"), default="rel")
    a = ap.parse_args()
    rep = build_map_plus(a.log, a.out, a.max_eps, not a.no_contact, depth=a.depth)
    print(json.dumps({k: rep[k] for k in ("frames_accepted", "frames_total", "occupied", "contact_columns",
                                          "contact_voxels", "contact_classes")}, ensure_ascii=False))
