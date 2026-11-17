"""Експорт усіх успішних обльотів натурного набору в журнали АС Р1.

  python export_field.py ../data/field_20261104 --out ../runs/real_20261104

Береться кожен запис survey з результатом «ok» і повним набором кадрів зйомки. Калібрування --
calib/*.json набору з тим самим серійним номером, положення -- кроки з курсом із рискання.
Пише <out>/<сцена>_d<номер>_<час>/ (meta.json, telemetry.csv, frames/) і <out>/index.csv.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postprocess as pp   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="тека розпакованого results_*.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", choices=("imu", "commands"), default="imu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for run in sorted(glob.glob(os.path.join(a.root, "runs", "*survey*"))):
        meta = json.load(open(os.path.join(run, "meta.json"), encoding="utf-8"))
        mission = json.load(open(os.path.join(run, "mission.json"), encoding="utf-8"))
        marks = json.load(open(os.path.join(run, "marks.json"), encoding="utf-8"))
        caps = sum(1 for m in marks if m["kind"] == "capture")
        want = len(mission["waypoints"]) * int(mission.get("turns", 4))
        if meta.get("outcome") != "ok" or caps != want:
            continue
        drone = meta.get("drone") or {}
        sn = (drone.get("info") or {}).get("sn") or drone.get("sn")
        calib = pp.find_calib(run, os.path.join(a.root, "calib"))
        name = f"{mission.get('scene', 'S')}_d{drone.get('label')}_{os.path.basename(run)[:15]}"
        out = os.path.join(a.out, name)
        t0 = time.time()
        res = pp.export_ac1(run, out, calib=calib, positions=a.positions, verbose=False)
        log = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
        rows.append({"log": name, "run": os.path.basename(run), "scene": mission.get("scene"),
                     "drone": drone.get("label"), "sn": sn, "frames": res["frames"], "captures": caps,
                     "calib": os.path.basename(calib) if calib else "", "hfov_deg": log["hfov_deg"],
                     "yaw_sign": log["yaw_sign"], "positions": a.positions,
                     "software": meta.get("software", "")})
        print(f"{name}: {res['frames']} кадрів, калібрування {rows[-1]['calib'] or 'НЕМАЄ'}, "
              f"кут огляду {log['hfov_deg']}°, {time.time() - t0:.0f} с")
    with open(os.path.join(a.out, "index.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Експортовано {len(rows)} обльотів -> {a.out}")


if __name__ == "__main__":
    main()
