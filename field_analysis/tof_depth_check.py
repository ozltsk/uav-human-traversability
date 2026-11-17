"""Масштаб монокулярної глибини АС Р1 проти переднього ToF RoboMaster TT (незалежний еталон, до 1,25 м).

  python tof_depth_check.py ../data/field_20261104 --out ../runs/real_20261104/tof_check

Для кожного показу переднього ToF 0,15-1,25 м у польотах (line, survey): найближчий кадр (до 0,25 с),
нейромережі АС Р1 (глибина + підлога), масштаб за підлогою з висотою з нижнього ToF, глибина
в околі пікселя горизонтального променя (ToF дивиться вперед горизонтально, камера -- на tilt униз).
Перебираються нахил камери й поправка висоти: яка пара дає відстань, найближчу до ToF.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac1_evacmap"))
sys.path.insert(0, os.path.join(ROOT, "rmtt_field"))
from mono_depth import CameraModel, floor_depth, recover_scale, metric_depth   # noqa: E402
import postprocess as pp                                                     # noqa: E402

TILTS = (0.0, 5.0, 10.0, 15.0, 20.0)
DH = (0.0, -0.1, -0.2, -0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from perception import Perception
    perc = Perception(device="cpu")
    rows = []
    for run in sorted(glob.glob(os.path.join(a.root, "runs", "*"))):
        name = os.path.basename(run)
        if not ("survey" in name or "line" in name) or not os.path.exists(os.path.join(run, "ext_tof.csv")):
            continue
        tofs = [(float(r["t"]), int(r["tof_mm"])) for r in csv.DictReader(open(os.path.join(run, "ext_tof.csv"), encoding="utf-8"))
                if r["tof_mm"] and 150 < int(r["tof_mm"]) < 1250]
        if not tofs:
            continue
        ft = pp.frame_times(run)
        if ft is None:
            continue
        times = ft["times"]
        st = pp.load_state(run)
        want = {}
        for t, mm in tofs:
            i = int(np.argmin(np.abs(times - t)))
            if abs(times[i] - t) <= 0.25:
                want[i] = (t, mm)
        if not want:
            continue
        calib = pp.find_calib(run, os.path.join(a.root, "calib"))
        K = np.array(json.load(open(calib, encoding="utf-8"))["K"]) if calib else None
        for i, _t, frame in pp.iter_frames(run, ft):
            if i not in want:
                continue
            t, mm = want.pop(i)
            h, w = frame.shape[:2]
            hfov = float(np.degrees(2 * np.arctan(w / (2 * K[0, 0])))) if K is not None else 55.5
            rel, mask = perc.predict(frame[:, :, ::-1].copy())
            z_tof = float(pp.interp_state(st, "tof", t)) / 100.0
            res = {"run": name, "frame": i, "tof_m": mm / 1000.0, "z_tof": round(z_tof, 3),
                   "floor_frac": round(float(mask.mean()), 3)}
            for tilt in TILTS:
                cam = CameraModel(w, h, hfov, tilt)
                f = cam.fx
                v = int(round(h / 2 - f * np.tan(np.radians(tilt))))
                for dh in DH:
                    fit = recover_scale(rel, mask, floor_depth(cam, 0.0, z_tof + dh), rng=np.random.default_rng(1))
                    key = f"t{tilt:g}_h{dh:+.1f}"
                    if not fit["accepted"]:
                        res[key] = None
                        continue
                    Z = metric_depth(rel, fit["alpha"], fit["beta"])
                    win = Z[max(0, v - 12):v + 13, w // 2 - 12:w // 2 + 13]
                    D = float(np.nanmedian(win)) / np.cos(np.radians(tilt))
                    res[key] = round(D, 3)
            rows.append(res)
            print(f"{name[:40]} кадр {i}: ToF {mm} мм, висота {z_tof:.2f} м, підлога {res['floor_frac']:.2f}, "
                  f"моно (10°, 0) {res.get('t10_h+0.0')} м")
            if not want:
                break
    with open(os.path.join(a.out, "tof_vs_mono.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\nвимірів {len(rows)}; відношення моно / ToF (медіана [25 %..75 %], кадрів з масштабом):")
    for tilt in TILTS:
        line = []
        for dh in DH:
            key = f"t{tilt:g}_h{dh:+.1f}"
            r = np.array([x[key] / x["tof_m"] for x in rows if x.get(key)])
            line.append(f"h{dh:+.1f}: {np.median(r):.2f} [{np.percentile(r, 25):.2f}..{np.percentile(r, 75):.2f}] n={len(r)}"
                        if len(r) else f"h{dh:+.1f}: --")
        print(f"  нахил {tilt:4.0f}°: " + " | ".join(line))


if __name__ == "__main__":
    main()
