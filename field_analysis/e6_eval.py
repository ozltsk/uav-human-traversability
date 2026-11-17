# -*- coding: utf-8 -*-
"""e6_eval.py -- підтверджувальна серія E6 (11-16.11.2026) зафіксованим конвеєром (drones2026/E6_frozen.json).

Нічого не підбирає: карта -- ac1_evacmap buildmap --max-eps 0.012; стани, маршрут і перевірка безпеки --
ті самі функції, що в route_safety2.py (theta 0,8; kappa 0 для М3 і 6 для решти). Відмінності від
field_eval.py лише в даних сцени: точки start/goal, предмети й проходи читаються з ground_truth E6.
Сцени S9 і S10 зняті двома польотами (A і B): кадри обох польотів одного дрона складаються в один журнал
у спільних координатах кімнати, далі той самий buildmap.

    python -X utf8 e6_eval.py [--skip-build]
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac2_evacroute"))
sys.path.insert(0, HERE)
import route_safety2 as rs2                                              # noqa: E402
from field_eval import read_csv, num, band_cells, expected_state, STATE_UA   # noqa: E402
from pose_model import WALK, BLOCKED, UNSEEN                             # noqa: E402
from nn_traversability import load_model                                 # noqa: E402
from route_astar import plan_route, cells_near                           # noqa: E402

RUNS = os.path.join(ROOT, "runs", "real_20261116")
GT = os.path.join(ROOT, "data", "field_20261116", "ground_truth")
OUT = os.path.join(RUNS, "eval_e6")
MAX_EPS, THETA = 0.012, 0.8
METHODS = ("М3", "М4", "М5", "М5р")
SCENES = ("S6", "S7", "S8", "S9", "S10")


def gt_objects(scene):
    out = []
    for o in read_csv(os.path.join(GT, f"{scene}_objects.csv")):
        x0, y0, x1, y1 = (num(o.get(k, "")) for k in ("x0", "y0", "x1", "y1"))
        if None in (x0, y0, x1, y1):
            continue
        out.append((o["object_id"], min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), num(o.get("clearance_m", ""))))
    return out


def gt_passages(scene, voxel, shape):
    pas, bypass = [], []
    for p in read_csv(os.path.join(GT, f"{scene}_passages.csv")):
        if num(p.get("y0")) is None:
            continue
        seg = (num(p["x0"]), num(p["y0"]), num(p["x1"]), num(p["y1"]))
        d = {"id": p["passage_id"], "width": num(p.get("width_m")), "clearance": num(p.get("clearance_m")),
             "cells": set(band_cells(seg, voxel, shape))}
        d["expected"] = expected_state(d["width"], d["clearance"])
        (bypass if p["passage_id"].upper().startswith("BYPASS") else pas).append(d)
    return pas, bypass


def route_points(scene):
    pts = {r["point"].strip().lower(): (num(r["x_m"]), num(r["y_m"])) for r in read_csv(os.path.join(GT, f"{scene}_route.csv"))}
    return pts["start"], pts["goal"]


def merge_logs(a_dir, b_dir, out_dir):
    """Один журнал із двох польотів: телеметрія й прогнози мережі B дописуються після A (індекси +1000)."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "pred"))
    meta = json.load(open(os.path.join(a_dir, "meta.json"), encoding="utf-8"))
    meta["merged_from"] = [os.path.basename(a_dir), os.path.basename(b_dir)]
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rows = []
    for d, off in ((a_dir, 0), (b_dir, 1000)):
        for r in csv.DictReader(open(os.path.join(d, "telemetry.csv"), newline="", encoding="utf-8")):
            src = os.path.join(d, "pred", f"pred_{int(r['index']):04d}.npz")
            r["index"] = int(r["index"]) + off
            r["waypoint"] = int(r["waypoint"]) + off // 10
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out_dir, "pred", f"pred_{r['index']:04d}.npz"))
            rows.append(r)
    with open(os.path.join(out_dir, "telemetry.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "x", "y", "z_tof", "yaw_deg", "waypoint"])
        w.writeheader()
        w.writerows(rows)


def units():
    """Одиниці оцінювання: обліт (S6-S8) або пара A+B одного дрона (S9, S10). first -- перший ok на дрон і сцену."""
    idx = list(csv.DictReader(open(os.path.join(RUNS, "index.csv"), encoding="utf-8-sig")))
    by = defaultdict(list)
    for r in idx:
        m = re.match(r"(S\d+)([AB]?)$", r["scene"])
        if not m or m.group(1) not in SCENES:
            continue
        by[(m.group(1), r["sn"], m.group(2))].append(r["log"])
    out = []
    for (scene, sn, side), logs in sorted(by.items()):
        logs.sort(key=lambda s: s[-15:])
        if side == "":
            for k, l in enumerate(logs):
                out.append({"name": l, "scene": scene, "sn": sn, "logs": [l], "first": k == 0})
        elif side == "A":
            bs = sorted(by.get((scene, sn, "B"), []), key=lambda s: s[-15:])
            for ka, la in enumerate(logs):
                for kb, lb in enumerate(bs):
                    out.append({"name": f"{scene}_{sn[-3:]}_{la[-15:]}+{lb[-15:]}", "scene": scene, "sn": sn,
                                "logs": [la, lb], "first": ka == 0 and kb == 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    model = load_model(os.path.join(ROOT, "runs", "ac2_models", "travnet.pt"), "cpu")
    model_r = load_model(os.path.join(ROOT, "runs", "ac2_models_rmtt", "travnet.pt"), "cpu")
    res = []
    for u in units():
        npz = os.path.join(OUT, u["name"] + ".npz")
        if not (a.skip_build and os.path.exists(npz)):
            if len(u["logs"]) == 2:
                log_dir = os.path.join(RUNS, "_merged", u["name"])
                merge_logs(os.path.join(RUNS, u["logs"][0]), os.path.join(RUNS, u["logs"][1]), log_dir)
            else:
                log_dir = os.path.join(RUNS, u["logs"][0])
            subprocess.run([sys.executable, os.path.join(ROOT, "ac1_evacmap", "app.py"), "buildmap", "--log", log_dir,
                            "--out", npz, "--max-eps", str(MAX_EPS)], check=True, stdout=subprocess.DEVNULL,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        rep = json.load(open(os.path.splitext(npz)[0] + "_summary.json", encoding="utf-8"))
        st, layers, voxel = rs2.states_for(npz, THETA, model, model_r)
        shape = st["М3"].shape
        scene = u["scene"]
        objs = gt_objects(scene)
        pas, bypass = gt_passages(scene, voxel, shape)
        s0, g0 = route_points(scene)
        start = (int(s0[0] / voxel), int(s0[1] / voxel))
        exits = cells_near(g0, voxel, 0.4, shape)
        truth_exists = any(p["expected"] != BLOCKED for p in pas + bypass)
        # контроль «порожня карта»: лише межі кімнати, усі клітинки прохідні пішки
        empty = np.full(shape, WALK, dtype=st["М3"].dtype)
        zeros = np.zeros(shape, dtype=float)
        cand = dict(st)
        row = {"name": u["name"], "scene": scene, "sn": u["sn"], "first": u["first"], "logs": u["logs"],
               "frames_accepted": rep.get("frames_accepted"), "frames_total": rep.get("frames_total"),
               "eps_median": rep.get("eps_median"), "truth_route_exists": truth_exists,
               "start_state": {m: STATE_UA[int(st[m][start])] for m in METHODS}, "methods": {}, "passages": {}}
        for m in METHODS + ("порожня",):
            if m == "порожня":
                s, sup, unk, kappa = empty, zeros, zeros, 6.0
            else:
                s, sup, unk, kappa = cand[m], layers["support"], layers["unknown_crawl"], (0.0 if m == "М3" else 6.0)
            path = plan_route(s, sup, unk, start, exits, voxel, kappa=kappa)
            c = rs2.classify2(scene, path, s, voxel, objs, pas)
            if path:
                pset = set(path)
                c["via"] = ("bypass" if any(pset & b["cells"] for b in bypass)
                            else (c["crossing"][0][0] if c.get("crossing") else "other")) if c["class"] == "safe" else c.get("via")
                c["path"] = [[int(i), int(j)] for i, j in path]
            c["correct"] = (c["class"] == "safe") if truth_exists else (c["class"] == "none")
            row["methods"][m] = c
        for p in pas + bypass:
            row["passages"][p["id"]] = {"expected": STATE_UA[p["expected"]]}
            for m in METHODS:
                vals = np.array([st[m][c] for c in sorted(p["cells"])])
                seen = vals[vals != UNSEEN]
                best = int(np.bincount(seen, minlength=5).argmax()) if len(seen) else UNSEEN
                row["passages"][p["id"]][m] = STATE_UA[best]
        res.append(row)
        print(u["name"], f"кадрів {row['frames_accepted']}/{row['frames_total']}",
              {m: row["methods"][m]["class"] + ("" if row["methods"][m]["class"] != "unsafe" else ": " + row["methods"][m]["reason"])
               for m in METHODS + ("порожня",)}, flush=True)
    json.dump(res, open(os.path.join(OUT, "e6_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for label, sel in (("усі ok", res), ("перший ok на дрон і сцену", [r for r in res if r["first"]])):
        print(f"\n== {label}: одиниць {len(sel)}")
        for m in METHODS + ("порожня",):
            c = Counter(r["methods"][m]["class"] for r in sel)
            ok = sum(r["methods"][m]["correct"] for r in sel)
            per = {s: f"{sum(r['methods'][m]['correct'] for r in sel if r['scene'] == s)}/{sum(r['scene'] == s for r in sel)}"
                   for s in SCENES}
            print(f"  {m:8} правильних {ok}/{len(sel)}  safe {c.get('safe', 0)} unsafe {c.get('unsafe', 0)} none {c.get('none', 0)}  {per}")


if __name__ == "__main__":
    main()
