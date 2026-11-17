# -*- coding: utf-8 -*-
"""posture_sensitivity.py -- how the route results of the rule-based estimators depend on the posture bands.

The reference bands are walk 1.75/0.50 m, crouch 1.20/0.50 m, crawl 0.50/0.60 m (height/width). Each variant changes
the bands in the passability rules AND in the posture-aware safety check, re-plans the routes of M3 and M4 on the 18
reference maps (floor-fit threshold 0.012) and counts safe / unsafe / none. Also: per-hover-point offsets between the
dead-reckoned positions and the positions solved from the floor crosses (control variant, five surveys).

    python posture_sensitivity.py  ->  ../runs/real_20261104/posture_sensitivity.json
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac2_evacroute"))
sys.path.insert(0, HERE)
import pose_model                                                     # noqa: E402
import route_safety2 as RS                                            # noqa: E402
from map_layers import load_map_npz, column_layers                    # noqa: E402
from route_astar import plan_route, cells_near                        # noqa: E402
from field_eval import START, EXIT                                    # noqa: E402

RUNS = RS.RUNS
EV = "eval18_eps012_th08"
REF = (("walk", 1.75, 0.5, 1.2), ("crouch", 1.2, 0.5, 0.5), ("crawl", 0.5, 0.6, 0.25))
VARIANTS = {
    "reference": REF,
    "walk 1.90 m": (("walk", 1.90, 0.5, 1.2), REF[1], REF[2]),
    "widths +0.10 m": (("walk", 1.75, 0.6, 1.2), ("crouch", 1.2, 0.6, 0.5), ("crawl", 0.5, 0.7, 0.25)),
    "crawl 0.60 m high": (REF[0], REF[1], ("crawl", 0.6, 0.6, 0.25)),
}


def set_poses(p):
    pose_model.POSES = p
    RS.POSES = p


def main():
    rep = json.load(open(os.path.join(RUNS, EV, "report.json"), encoding="utf-8"))
    out = {"variants": {}, "positions": {}}
    for tag, poses in VARIANTS.items():
        set_poses(poses)
        c = {"М3": Counter(), "М4": Counter()}
        for r in rep:
            state3d, voxel, eps = load_map_npz(os.path.join(RUNS, EV, r["log"] + ".npz"))
            layers = column_layers(state3d, voxel)
            shape = layers["support"].shape
            objs, pas = RS.gt_objects(r["scene"]), RS.gt_passages(r["scene"], voxel, shape)
            start = (int(START[0] / voxel), int(START[1] / voxel))
            exits = cells_near(EXIT, voxel, 0.4, shape)
            for m, mu, umax, kappa in (("М3", 0.0, 1.0, 0.0), ("М4", pose_model.safety_margin(eps if eps > 0 else 0.03), 0.9, 6.0)):
                st = pose_model.rule_state_map(layers, voxel, mu=mu, u_max=umax)
                path = plan_route(st, layers["support"], layers["unknown_crawl"], start, exits, voxel, kappa=kappa)
                c[m][RS.classify2(r["scene"], path, st, voxel, objs, pas)["class"]] += 1
        out["variants"][tag] = {m: dict(v) for m, v in c.items()}
        print(tag, out["variants"][tag], flush=True)
    set_poses(REF)
    # intermediate hover points: dead reckoning vs the floor-cross solution
    sel = [s.strip() for s in open(os.path.join(RUNS, "_selected.txt"), encoding="utf-8") if s.strip()]
    offs = []
    for name in sel:
        a = list(csv.DictReader(open(os.path.join(RUNS, name, "telemetry.csv"), encoding="utf-8")))
        b = list(csv.DictReader(open(os.path.join(RUNS, name + "_xy", "telemetry.csv"), encoding="utf-8")))
        per = {}
        for ra, rb in zip(a, b):
            per.setdefault(int(ra["waypoint"]), []).append(float(np.hypot(float(rb["x"]) - float(ra["x"]), float(rb["y"]) - float(ra["y"]))))
        per = {wp: float(np.median(v)) for wp, v in per.items()}
        out["positions"][name] = per
        offs += [v for v in per.values() if v > 1e-6]          # 0 = point left uncorrected (too few crosses seen)
    out["positions_summary"] = {"n_points_corrected": len(offs), "n_points_total": 4 * len(sel),
                                "min": min(offs), "median": float(np.median(offs)), "max": max(offs)}
    print(out["positions_summary"])
    json.dump(out, open(os.path.join(RUNS, "posture_sensitivity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
