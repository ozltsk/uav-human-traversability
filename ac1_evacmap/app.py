"""Нейромережеве картографування закритого приміщення за відеоданими
малого БПЛА з відновленням метричного масштабу.

Команди:
  survey    -- обліт приміщення дроном DJI Tello, запис журналу обльоту
  simflight -- синтетичний обліт процедурно згенерованого приміщення
  predict   -- обробка кадрів журналу нейромережами (глибина + підлога)
  buildmap  -- відновлення масштабу, фільтрація та побудова 3D-карти
  frame     -- обробка одного зображення (демонстрація сприйняття)
  show      -- рисунки для збереженої карти
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from flight_log import FlightLog
from mono_depth import CameraModel, process_frame, floor_depth, recover_scale, metric_depth, flying_pixel_mask
from occupancy_map import OccupancyMap
import visualize as viz


def camera_from_meta(meta):
    w, h = meta.get("frame_size", [960, 720])
    return CameraModel(int(w), int(h), float(meta.get("hfov_deg", 70.0)), float(meta.get("tilt_deg", 10.0)))


# ---------------------------------------------------------------- buildmap
def build_map(log, voxel=0.1, tau=0.06, rho=0.2, gamma=0.08, stride=2, max_range=4.0,
              room=None, verbose=True, seed=0, max_eps=None):
    """max_eps -- необов'язковий поріг похибки підгонки підлоги: кадр з eps > max_eps відкидається
    (натурні обльоти RMTT, 04.11.2026: дерев'яну панель на стіні сегментатор брав за підлогу, і
    масштаб виходив у 2-3 рази завеликим при eps ~0,02; у звичайних кадрах eps ~0,006)."""
    cam = camera_from_meta(log.meta)
    room = room or log.meta.get("room_size", [12.0, 12.0, 3.0])
    occ = OccupancyMap(size_xyz=tuple(room), voxel=voxel, max_range=max_range)
    rng = np.random.default_rng(seed)
    rows = []
    t_start = time.perf_counter()
    for rec in log.records:
        if not log.has_pred(rec.index):
            continue
        rel, mask = log.load_pred(rec.index)
        t0 = time.perf_counter()
        fit, Z, pts = process_frame(cam, rel, mask, rec.z_tof, rec.yaw_deg,
                                    (rec.x, rec.y, rec.z_tof), tau=tau, rho=rho, gamma=gamma,
                                    stride=stride, rng=rng)
        n_pts = 0
        if fit["accepted"] and max_eps is not None and fit["eps"] > max_eps:
            fit = {**fit, "accepted": False}
        if fit["accepted"]:
            n_pts = occ.integrate_points((rec.x, rec.y, rec.z_tof), pts, eps=fit["eps"])
        dt = time.perf_counter() - t0
        row = {"index": rec.index, "waypoint": rec.waypoint, "yaw_deg": rec.yaw_deg, "z_tof": rec.z_tof,
               "accepted": bool(fit["accepted"]), "n_floor": fit["n_floor"], "n_inliers": fit["n_inliers"],
               "eps": fit["eps"], "alpha": fit["alpha"], "beta": fit["beta"], "points": n_pts, "time_s": dt}
        rows.append(row)
        if verbose:
            if fit["accepted"]:
                print(f"кадр {rec.index:4d} (точка {rec.waypoint:2d}, {rec.yaw_deg:6.1f} град): підлога {fit['n_floor']:6d} пікс., "
                      f"інлаєрів {fit['n_inliers']:6d}, eps = {fit['eps']:.3f}, точок у карту {n_pts:6d}, {dt:.2f} с")
            else:
                print(f"кадр {rec.index:4d} (точка {rec.waypoint:2d}, {rec.yaw_deg:6.1f} град): ВІДКИНУТО "
                      f"(підлога {fit['n_floor']} пікс., інлаєрів {fit['n_inliers']})")
    total = time.perf_counter() - t_start
    if verbose:
        s = occ.summary()
        print(f"Кадрів оброблено {len(rows)}, прийнято {sum(r['accepted'] for r in rows)}, час {total:.1f} с")
        print(f"Вокселів: зайнятих {s['occupied']}, вільних {s['free']}, невідомих {s['unknown']}; "
              f"медіана eps = {s['eps_median']}")
    return occ, rows


def write_report(rows, occ, path, extra=None):
    with open(path + "_frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rep = occ.summary()
    rep["frames_accepted"] = int(sum(r["accepted"] for r in rows))
    rep["frames_total"] = len(rows)
    if extra:
        rep.update(extra)
    with open(path + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return rep


# ---------------------------------------------------------------- команди
def cmd_survey(a):
    from drone_survey import TelloSurvey, load_waypoints, grid_waypoints
    if a.waypoints:
        wps = load_waypoints(a.waypoints)
    else:
        wps = grid_waypoints((0.0, a.grid_x), (0.0, a.grid_y), a.spacing)
    s = TelloSurvey(a.out, height=a.height)
    s.connect()
    try:
        log = s.survey(wps, turns=4)
        print(f"Записано {len(log)} кадрів у {a.out}")
    finally:
        s.disconnect()


def cmd_simflight(a):
    from sim_flight import simulate_flight
    t0 = time.perf_counter()
    log, scene = simulate_flight(a.out, seed=a.seed, level=a.level, width=a.width, height=a.height_px,
                                 per_zone=a.per_zone)
    print(f"Сцена {a.seed}: боксів {len(scene['boxes'])}, прорізів {len(scene['openings'])}, "
          f"точок зависання {len(log.meta['waypoints'])}, кадрів {len(log)}, {time.perf_counter() - t0:.1f} с")
    for o in scene["openings"]:
        print(f"  проріз x={o['x']:.0f} м, y={o['y0']:.2f}-{o['y1']:.2f} м: {o['kind']}, висота {o['height']:.2f} м")


def cmd_predict(a):
    from perception import Perception, run_perception_on_log
    log = FlightLog.load(a.log)
    p = Perception(device=a.device)
    n = run_perception_on_log(log, p, overwrite=a.overwrite)
    print(f"Оброблено кадрів: {n}")


def cmd_buildmap(a):
    log = FlightLog.load(a.log)
    occ, rows = build_map(log, voxel=a.voxel, tau=a.tau, rho=a.rho, gamma=a.gamma, stride=a.stride,
                          max_range=a.max_range, max_eps=a.max_eps)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    occ.save(a.out, extra={"source": log.root})
    extra = {}
    truth_path = os.path.join(a.log, "truth_occupancy.npz")
    if os.path.exists(truth_path):
        truth = np.load(truth_path)["occupancy"]
        extra["vs_truth"] = occ.compare_with_truth(truth)
        print("Порівняння з точною сценою:", extra["vs_truth"])
    rep = write_report(rows, occ, os.path.splitext(a.out)[0], extra)
    if a.figs:
        os.makedirs(a.figs, exist_ok=True)
        truth = np.load(truth_path)["occupancy"] if os.path.exists(truth_path) else None
        viz.map_top_view(occ, log, truth=truth, out=os.path.join(a.figs, "map_top.png"))
        viz.map_slices(occ, out=os.path.join(a.figs, "map_slices.png"))
        viz.map_3d(occ, out=os.path.join(a.figs, "map_3d.png"))
        viz.frame_table_plot(rows, out=os.path.join(a.figs, "frames_eps.png"))
        cam = camera_from_meta(log.meta)
        for idx in a.panel_frames:
            rec = log.records[idx]
            rel, mask = log.load_pred(rec.index)
            z_floor = floor_depth(cam, rec.yaw_deg, rec.z_tof)
            fit = recover_scale(rel, mask, z_floor, tau=a.tau, rho=a.rho, rng=np.random.default_rng(1))
            Z = metric_depth(rel, fit["alpha"], fit["beta"]) if fit["accepted"] else None
            keep = flying_pixel_mask(Z, a.gamma) if Z is not None else None
            img = log.load_frame(rec.index) if log.has_frame(rec.index) else np.zeros(rel.shape + (3,), np.uint8)
            viz.frame_panel(img, rel, mask, Z, fit, keep,
                            title=f"Кадр {rec.index}: точка {rec.waypoint}, висота {rec.z_tof:.2f} м, рискання {rec.yaw_deg:.0f} град",
                            out=os.path.join(a.figs, f"frame_{rec.index:04d}.png"))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def cmd_frame(a):
    from PIL import Image
    from perception import Perception
    img = np.asarray(Image.open(a.image).convert("RGB"))
    p = Perception(device=a.device)
    rel, mask = p.predict(img)
    t = p.last_timing
    print(f"Глибина {t['depth_s']:.2f} с, сегментація {t['seg_s']:.2f} с; підлога {100.0 * mask.mean():.2f} % пікселів")
    cam = CameraModel(img.shape[1], img.shape[0], a.hfov, a.tilt)
    z_floor = floor_depth(cam, a.yaw, a.height)
    fit = recover_scale(rel, mask, z_floor, rng=np.random.default_rng(1))
    Z = metric_depth(rel, fit["alpha"], fit["beta"]) if fit["accepted"] else None
    keep = flying_pixel_mask(Z) if Z is not None else None
    print("Результат відновлення масштабу:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fit.items()})
    viz.frame_panel(img, rel, mask, Z, fit, keep, title=a.title or os.path.basename(a.image), out=a.out)
    print("Рисунок збережено:", a.out)


def cmd_show(a):
    occ = OccupancyMap.load(a.map)
    log = FlightLog.load(a.log) if a.log else None
    os.makedirs(a.figs, exist_ok=True)
    viz.map_top_view(occ, log, out=os.path.join(a.figs, "map_top.png"))
    viz.map_slices(occ, out=os.path.join(a.figs, "map_slices.png"))
    viz.map_3d(occ, out=os.path.join(a.figs, "map_3d.png"))
    print(occ.summary())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("survey", help="обліт дроном DJI Tello")
    s.add_argument("--out", required=True)
    s.add_argument("--waypoints", help="CSV з колонками x, y (м)")
    s.add_argument("--grid-x", type=float, default=4.0)
    s.add_argument("--grid-y", type=float, default=4.0)
    s.add_argument("--spacing", type=float, default=2.0)
    s.add_argument("--height", type=float, default=1.3)
    s.set_defaults(func=cmd_survey)

    s = sub.add_parser("simflight", help="синтетичний обліт")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--level", type=int, default=2, help="рівень похибки нейромережі 0..4")
    s.add_argument("--width", type=int, default=320)
    s.add_argument("--height-px", type=int, default=240)
    s.add_argument("--per-zone", type=int, default=4)
    s.set_defaults(func=cmd_simflight)

    s = sub.add_parser("predict", help="нейромережева обробка кадрів журналу")
    s.add_argument("--log", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("buildmap", help="побудова 3D-карти з журналу")
    s.add_argument("--log", required=True)
    s.add_argument("--out", required=True, help="файл карти .npz")
    s.add_argument("--figs", help="тека для рисунків")
    s.add_argument("--voxel", type=float, default=0.1)
    s.add_argument("--tau", type=float, default=0.06)
    s.add_argument("--rho", type=float, default=0.2)
    s.add_argument("--gamma", type=float, default=0.08)
    s.add_argument("--stride", type=int, default=2)
    s.add_argument("--max-range", type=float, default=4.0)
    s.add_argument("--max-eps", type=float, default=None, help="відкидати кадри з похибкою підлоги, більшою за поріг")
    s.add_argument("--panel-frames", type=int, nargs="*", default=[0, 5])
    s.set_defaults(func=cmd_buildmap)

    s = sub.add_parser("frame", help="обробка одного зображення")
    s.add_argument("--image", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--height", type=float, default=1.3)
    s.add_argument("--yaw", type=float, default=0.0)
    s.add_argument("--hfov", type=float, default=70.0)
    s.add_argument("--tilt", type=float, default=10.0)
    s.add_argument("--device", default="cpu")
    s.add_argument("--title")
    s.set_defaults(func=cmd_frame)

    s = sub.add_parser("show", help="рисунки для збереженої карти")
    s.add_argument("--map", required=True)
    s.add_argument("--log")
    s.add_argument("--figs", required=True)
    s.set_defaults(func=cmd_show)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
