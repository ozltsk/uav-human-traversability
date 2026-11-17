"""Місія облету (survey) для сцени з перевіркою безпеки траєкторії.

  python make_mission.py --scene S2 --room 7.2 5.8 --origin 1.0 1.0 \
      --waypoints "1.0,1.0 3.0,1.0 3.0,4.6 1.0,4.6" --gt ground_truth/S2_objects.csv

Система приміщення: x уздовж обраної стіни, y ліворуч від неї, метри; початок -- кут приміщення.
Перевіряється: точки й відрізки між ними (від точки зльоту) не ближче --wall-margin до стін і
--obj-margin до предметів з ground_truth/<сцена>_objects.csv, дрон ніколи не пролітає над предметом
(Tello тримає висоту над тим, що під ним, і над столом підскочив би). Помилка -- місія не пишеться.
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_points(text):
    pts = []
    for tok in text.replace(";", " ").split():
        x, y = tok.split(",")
        pts.append((float(x), float(y)))
    return pts


def load_objects(path):
    """CSV з комами або (Excel з українськими налаштуваннями) з ';' і десятковою комою."""
    objs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delim = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    for row in csv.DictReader(text.splitlines(), delimiter=delim):
        if not (row.get("x0") or "").strip():
            continue
        x0, y0, x1, y1 = (float(row[k].strip().replace(",", ".")) for k in ("x0", "y0", "x1", "y1"))
        objs.append({"id": row.get("object_id", "?"), "type": row.get("type", ""),
                     "rect": (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))})
    return objs


def point_rect_dist(p, r):
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return float(np.hypot(dx, dy))


def segment_rect_dist(a, b, r, step=0.02):
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
    return min(point_rect_dist((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), r)
               for t in np.linspace(0.0, 1.0, n))


def validate(room, origin, wps, objs, wall_margin, obj_margin):
    errors, notes = [], []
    path = [tuple(origin)] + [tuple(p) for p in wps]
    for name, p in [("зліт", origin)] + [(f"точка {i + 1}", p) for i, p in enumerate(wps)]:
        dw = min(p[0], p[1], room[0] - p[0], room[1] - p[1])
        if dw < wall_margin:
            errors.append(f"{name} {p}: до стіни {dw:.2f} м < {wall_margin} м")
        for o in objs:
            d = point_rect_dist(p, o["rect"])
            if d < obj_margin:
                errors.append(f"{name} {p}: до предмета {o['id']} ({o['type']}) {d:.2f} м < {obj_margin} м")
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        seg = f"відрізок {'зліт' if i == 0 else f'точка {i}'} -> точка {i + 1}"
        if 0 < length < 0.2:
            errors.append(f"{seg}: {length:.2f} м -- менше за 0,2 м (Tello не виконає), об'єднайте точки")
        for o in objs:
            d = segment_rect_dist(a, b, o["rect"])
            if d < obj_margin:
                what = "ПРОЛІТ НАД ПРЕДМЕТОМ" if d == 0 else f"{d:.2f} м < {obj_margin} м"
                errors.append(f"{seg}: до предмета {o['id']} ({o['type']}) {what}")
        notes.append(f"{seg}: {length:.2f} м")
    return errors, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--room", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--origin", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--waypoints", required=True, help='"x1,y1 x2,y2 ..."')
    ap.add_argument("--heading", type=float, default=0.0, help="куди дивиться камера на старті, градуси від осі x")
    ap.add_argument("--height", type=float, default=1.2)
    ap.add_argument("--gt", default=None, help="ground_truth/<сцена>_objects.csv")
    ap.add_argument("--yaw-sign", type=int, default=-1)
    ap.add_argument("--wall-margin", type=float, default=0.6)
    ap.add_argument("--obj-margin", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    wps = parse_points(a.waypoints)
    objs = load_objects(a.gt) if a.gt else []
    if not a.gt:
        print("УВАГА: без --gt предмети не перевіряються")
    errors, notes = validate(a.room, a.origin, wps, objs, a.wall_margin, a.obj_margin)
    for n in notes:
        print("  " + n)
    if errors:
        print("МІСІЮ НЕ ЗАПИСАНО:")
        for e in errors:
            print("  - " + e)
        raise SystemExit(1)
    mission = {
        "type": "survey", "name": f"survey_{a.scene}", "scene": a.scene,
        "room_size": [a.room[0], a.room[1], 3.0], "origin": list(a.origin), "heading_deg": a.heading,
        "waypoints": [list(p) for p in wps], "height_m": a.height, "speed_cm_s": 30, "turns": 4,
        "settle_s": 1.5, "capture_hold_s": 0.6,
        "camera": {"hfov_deg": 55.5, "tilt_deg": 10.0, "yaw_sign": a.yaw_sign, "offset_m": 0.0,
                   "calib": None, "undistort": True},   # калібрування свого дрона -- під час експорту
        "safety": {"min_clear_m": 0.5, "on_block": "land"},
        "ground_truth": os.path.relpath(a.gt, HERE) if a.gt else None,
    }
    out = a.out or os.path.join(HERE, "missions", f"survey_{a.scene}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mission, f, ensure_ascii=False, indent=2)
    print(f"Місію записано: {out} ({len(wps)} точок, {len(wps) * 4} кадрів зйомки, предметів перевірено {len(objs)})")


if __name__ == "__main__":
    main()
