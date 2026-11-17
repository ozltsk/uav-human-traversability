"""Уточнення положень точок облету зіставленням їхніх локальних хмар точок (2D ICP), без еталону.

  python pose_refine.py ../runs/real_20261104/S2_d2_20261103_140940 [--out <тека>]

Для кожної точки облету з її кадрів (масштаб за підлогою, як в АС Р1) береться хмара перешкод:
висота 0,25-1,9 м, до 3,5 м від дрона, сітка 5 см. Перша точка -- опора (вона за 0,3-2,2 м від
хреста зльоту, тож найточніша). Кожна наступна вирівнюється жорстким перетворенням (зсув +
поворот) до вже вирівняних, потім ще два проходи «кожна до всіх інших». Поправки понад 1 м або 15°
вважаються невдалим зіставленням і не застосовуються. Результат -- новий журнал АС Р1
<лог>_refined з уточненими x, y і рисканням (кадри й передбачення копіюються) та pose_refine.json.
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ac1_evacmap"))
from flight_log import FlightLog                      # noqa: E402
from mono_depth import CameraModel, process_frame     # noqa: E402

Z_LO, Z_HI, RANGE, GRID = 0.25, 1.9, 2.5, 0.05
MAX_SHIFT, MAX_ROT = 1.0, 15.0


def local_cloud(log, recs, cam, seed=0):
    """Хмара перешкод точки облету в системі, паралельній кімнатній, з початком у точці зависання."""
    rng = np.random.default_rng(seed)
    out = []
    for rec in recs:
        if not log.has_pred(rec.index):
            continue
        rel, mask = log.load_pred(rec.index)
        fit, _Z, pts = process_frame(cam, rel, mask, rec.z_tof, rec.yaw_deg, (0.0, 0.0, rec.z_tof),
                                     stride=2, rng=rng)
        if not fit["accepted"]:
            continue
        sel = (pts[:, 2] > Z_LO) & (pts[:, 2] < Z_HI) & (np.hypot(pts[:, 0], pts[:, 1]) < RANGE)
        out.append(pts[sel, :2])
    if not out:
        return np.zeros((0, 2))
    p = np.vstack(out)
    key = np.unique(np.round(p / GRID).astype(np.int64), axis=0)
    return key * GRID


def kabsch2d(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return R, cd - R @ cs


def icp(src, dst_tree, dst, iters=40, trim=0.7, rotation=False):
    """2D перетворення src -> dst (лише зсув, якщо rotation=False); повертає R, t, медіани до і після.
    Без повороту за замовчуванням: курс з IMU точний до 2-3° (мітки heading, 04.11.2026), а поворот
    з ICP на реальних хмарах виходив -10...-24° через похибку монокулярної глибини на дальніх стінах."""
    R, t = np.eye(2), np.zeros(2)
    d0, _ = dst_tree.query(src)
    for k in range(iters):
        cur = src @ R.T + t
        d, idx = dst_tree.query(cur)
        max_d = max(0.15, 0.6 * (1 - k / iters))
        keep = d < max_d
        if keep.sum() < 30:
            break
        thr = np.quantile(d[keep], trim)
        keep &= d <= thr
        if rotation:
            dR, dt = kabsch2d(cur[keep], dst[idx[keep]])
        else:
            dR, dt = np.eye(2), np.median(dst[idx[keep]] - cur[keep], axis=0)
        R, t = dR @ R, dR @ t + dt
        if abs(math.atan2(dR[1, 0], dR[0, 0])) < 1e-4 and np.hypot(*dt) < 1e-4:
            break
    d1, _ = dst_tree.query(src @ R.T + t)
    return R, t, float(np.median(d0)), float(np.median(d1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rotation", action="store_true", help="дозволити поворот (типово лише зсув)")
    ap.add_argument("--range", type=float, default=RANGE, help="найбільша відстань точок від дрона, м")
    a = ap.parse_args()
    globals()["RANGE"] = a.range
    log = FlightLog.load(a.log)
    w, h = log.meta.get("frame_size", [960, 720])
    cam = CameraModel(int(w), int(h), float(log.meta.get("hfov_deg", 55.5)), float(log.meta.get("tilt_deg", 10.0)))
    wps = sorted({r.waypoint for r in log.records})
    recs = {k: [r for r in log.records if r.waypoint == k] for k in wps}
    nominal = {k: np.array([recs[k][0].x, recs[k][0].y]) for k in wps}
    clouds = {k: local_cloud(log, recs[k], cam, seed=k) for k in wps}
    pose = {k: (nominal[k].copy(), 0.0) for k in wps}            # (положення, поправка рискання, град)

    def world(k, p=None, th=None):
        p = pose[k][0] if p is None else p
        th = pose[k][1] if th is None else th
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        return clouds[k] @ np.array([[c, -s], [s, c]]).T + p

    steps = []
    order = [(k, [j for j in wps if j < k]) for k in wps[1:]]
    for _ in range(2):
        order += [(k, [j for j in wps if j != k]) for k in wps[1:]]
    for k, refs in order:
        refs = [j for j in refs if len(clouds[j])]
        if len(clouds[k]) < 50 or not refs:
            continue
        dst = np.vstack([world(j) for j in refs])
        R, t, before, after = icp(world(k), cKDTree(dst), dst, rotation=a.rotation)
        new_p = R @ pose[k][0] + t
        dth = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        total_shift = float(np.hypot(*(new_p - nominal[k])))
        total_rot = pose[k][1] + dth
        ok = total_shift <= MAX_SHIFT and abs(total_rot) <= MAX_ROT and after <= before
        steps.append({"waypoint": k, "refs": refs, "median_before": round(before, 3), "median_after": round(after, 3),
                      "shift_m": round(total_shift, 3), "rot_deg": round(total_rot, 2), "applied": bool(ok)})
        if ok:
            pose[k] = (new_p, total_rot)
    out = a.out or a.log.rstrip("/\\") + "_refined"
    os.makedirs(out, exist_ok=True)
    for sub in ("frames", "pred"):
        if os.path.isdir(os.path.join(a.log, sub)):
            shutil.copytree(os.path.join(a.log, sub), os.path.join(out, sub), dirs_exist_ok=True)
    meta = dict(log.meta)
    result = {k: {"nominal": [round(float(v), 3) for v in nominal[k]],
                  "refined": [round(float(v), 3) for v in pose[k][0]],
                  "yaw_correction_deg": round(pose[k][1], 2), "cloud_points": int(len(clouds[k]))} for k in wps}
    meta["pose_refine"] = {"method": f"2D ICP між точками облету ({'зсув і поворот' if a.rotation else 'лише зсув'}, "
                                     f"до {RANGE} м), опора -- точка 1", "waypoints": result}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "telemetry.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "x", "y", "z_tof", "yaw_deg", "waypoint"])
        for r in log.records:
            p, th = pose[r.waypoint]
            yaw = (r.yaw_deg + th + 180.0) % 360.0 - 180.0
            wr.writerow([r.index, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{r.z_tof:.3f}", f"{yaw:.1f}", r.waypoint])
    with open(os.path.join(out, "pose_refine.json"), "w", encoding="utf-8") as f:
        json.dump({"waypoints": result, "steps": steps}, f, ensure_ascii=False, indent=1)
    shifts = ", ".join(f"т{k + 1}: {np.hypot(*(pose[k][0] - nominal[k])):.2f} м / {pose[k][1]:+.1f}°" for k in wps)
    print(f"{os.path.basename(a.log.rstrip('/'))}: {shifts}")


if __name__ == "__main__":
    main()
