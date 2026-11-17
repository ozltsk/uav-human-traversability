# -*- coding: utf-8 -*-
"""route_safety2.py -- posture-aware route safety check (review of 05.11.2026) and route variants.

A route is unsafe if any of its cells (a) lies inside the shrunk footprint of a measured object without clearance,
(b) lies under an object with measured clearance (table, pole) while the route uses a posture whose height band
exceeds that clearance, or (c) crosses a measured passage whose width is below the body-width requirement of the
posture used there (walk 0.50 m, crouch 0.50 m, crawl 0.60 m). The posture at a cell is the state the estimator gave
the cell, i.e. the one the planner used. The states are recomputed exactly as field_eval.py does, the routes are
taken from report.json (same cells), so the counts are comparable with the earlier footprint-only check.

Variants on the reference configuration: routes re-planned (same A*, same kappa) for two other start/goal pairs
and with the bypass along the wall with racks closed by a virtual wall (x 2.0-3.0 m, y below the barrier).

    python route_safety2.py [--out ../runs/real_20261104/route_safety2.json]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac2_evacroute"))
sys.path.insert(0, HERE)
from field_eval import read_csv, num, band_cells, START, EXIT                              # noqa: E402
from map_layers import load_map_npz, column_layers, features_from_layers                   # noqa: E402
from pose_model import rule_state_map, safety_margin, POSES, WALK, CROUCH, CRAWL, BLOCKED, UNSEEN  # noqa: E402
from nn_traversability import load_model, predict_probs, decide                            # noqa: E402
from route_astar import plan_route, cells_near                                             # noqa: E402
from aggregate_eval import BARRIER, SHRINK                                                 # noqa: E402

RUNS = os.path.join(ROOT, "runs", "real_20261104")
GT = os.path.join(ROOT, "data", "field_20261104", "ground_truth")
EVALS = {"eval18": 0.6, "eval18_eps012": 0.6, "eval18_eps012_th08": 0.8, "eval18_contact": 0.8,
         "eval5_base": 0.8, "eval5_metric_nothr": 0.8, "eval5_dense": 0.8, "eval5_dense_xy": 0.8}
REFERENCE = "eval18_eps012_th08"
METHODS = ("М3", "М4", "М5", "М5р")
PNAME = {WALK: "walk", CROUCH: "crouch", CRAWL: "crawl", BLOCKED: "blocked", UNSEEN: "unseen"}
PAIRS = {"A": (START, EXIT),                     # the paper's pair: behind the barrier -> door side
         "B": ((3.8, 4.8), (0.6, 4.8)),          # across the barrier near the desks: the bypass is a long detour
         "C": ((3.8, 4.0), (0.6, 1.0))}          # same start, goal at the corner of the wall with racks
WALL_X = (2.0, 3.0)                              # the virtual wall that closes the bypass
DESKS_Y = 5.25                                   # the desks start here (room depth 5.25 m)


def gt_objects(scene):
    out = []
    for o in read_csv(os.path.join(GT, f"{scene}_objects.csv")):
        x0, y0, x1, y1 = (num(o.get(k, "")) for k in ("x0", "y0", "x1", "y1"))
        if None in (x0, y0, x1, y1):
            continue
        out.append((o["object_id"], min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), num(o.get("clearance_m", ""))))
    # the row of desks along the far wall (y > 5.25 m, not measured piece by piece) is furniture, not a passage
    out.append(("desks", 0.0, DESKS_Y, 5.0, 6.0, None))
    return out


def gt_passages(scene, voxel, shape):
    out = []
    for p in read_csv(os.path.join(GT, f"{scene}_passages.csv")):
        if p["passage_id"] == "BYPASS" or num(p.get("y0")) is None:
            continue
        seg = (num(p["x0"]), num(p["y0"]), num(p["x1"]), num(p["y1"]))
        out.append({"id": p["passage_id"], "width": num(p.get("width_m")), "clearance": num(p.get("clearance_m")),
                    "cells": set(band_cells(seg, voxel, shape))})
    return out


def states_for(npz, theta, model, model_r):
    state3d, voxel, eps = load_map_npz(npz)
    layers = column_layers(state3d, voxel)
    feats = features_from_layers(layers)
    st = {"М3": rule_state_map(layers, voxel, mu=0.0, u_max=1.0),
          "М4": rule_state_map(layers, voxel, mu=safety_margin(eps if eps > 0 else 0.03), u_max=0.9),
          "М5": decide(predict_probs(model, feats, "cpu"), theta=theta, unknown_crawl=layers["unknown_crawl"]),
          "М5р": decide(predict_probs(model_r, feats, "cpu"), theta=theta, unknown_crawl=layers["unknown_crawl"])}
    return st, layers, voxel


def _inside(x, y, x0, y0, x1, y1):
    sx, sy = min(SHRINK, (x1 - x0) / 4), min(SHRINK, (y1 - y0) / 4)
    return x0 + sx <= x <= x1 - sx and y0 + sy <= y <= y1 - sy


def classify2(scene, path, state, voxel, objs, pas):
    """Posture-aware safety of one route. Returns a dict with class safe|unsafe|none, the reason, the passages crossed
    with the posture used, and how the barrier was passed (bypass | passage id | none)."""
    if not path:
        return {"class": "none"}
    crossing, postures = [], Counter()
    for (i, j) in path:
        x, y = (i + 0.5) * voxel, (j + 0.5) * voxel
        k = int(state[i, j])
        if k > CRAWL:                       # the planner never enters blocked or unseen cells; keep the check honest
            return {"class": "unsafe", "reason": "cell %s" % PNAME[k], "at": [round(x, 2), round(y, 2)]}
        pname, H, W, _v = POSES[k]
        postures[pname] += 1
        for oid, x0, y0, x1, y1, c in objs:
            if _inside(x, y, x0, y0, x1, y1):
                if c is None:
                    return {"class": "unsafe", "reason": "through object %s" % oid, "at": [round(x, 2), round(y, 2)]}
                if H > c + 1e-9:
                    return {"class": "unsafe", "reason": "%s under %s: clearance %.2f m < %.2f m" % (pname, oid, c, H),
                            "at": [round(x, 2), round(y, 2)]}
        for p in pas:
            if (i, j) in p["cells"]:
                if not crossing or crossing[-1][0] != p["id"]:
                    crossing.append([p["id"], pname])
                if p["width"] is not None and p["width"] < W - 1e-9:
                    return {"class": "unsafe", "reason": "%s through %s: width %.2f m < %.2f m" % (pname, p["id"], p["width"], W),
                            "at": [round(x, 2), round(y, 2)]}
                if p["clearance"] is not None and p["clearance"] < H - 1e-9:
                    return {"class": "unsafe", "reason": "%s through %s: clearance %.2f m < %.2f m" % (pname, p["id"], p["clearance"], H),
                            "at": [round(x, 2), round(y, 2)]}
    via = "no barrier"
    if scene in BARRIER:
        ys = [(j + 0.5) * voxel for i, j in path if WALL_X[0] <= (i + 0.5) * voxel <= WALL_X[1]]
        if ys and min(ys) < BARRIER[scene][0]:
            via = "bypass"
        elif crossing:
            via = crossing[0][0]
        else:
            via = "other"
    return {"class": "safe", "crossing": crossing, "via": via, "postures": dict(postures)}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS, "route_safety2.json"))
    ap.add_argument("--model", default=os.path.join(ROOT, "runs", "ac2_models", "travnet.pt"))
    ap.add_argument("--model-rmtt", default=os.path.join(ROOT, "runs", "ac2_models_rmtt", "travnet.pt"))
    a = ap.parse_args()
    model, model_r = load_model(a.model, "cpu"), load_model(a.model_rmtt, "cpu")
    out = {"evals": {}, "variants": {}, "summary": {}}
    for ev, theta in EVALS.items():
        rep_path = os.path.join(RUNS, ev, "report.json")
        if not os.path.exists(rep_path):
            continue
        rep = json.load(open(rep_path, encoding="utf-8"))
        res = {}
        for r in rep:
            name, scene = r["log"], r["scene"]
            st, layers, voxel = states_for(os.path.join(RUNS, ev, name + ".npz"), theta, model, model_r)
            shape = st["М3"].shape
            objs, pas = gt_objects(scene), gt_passages(scene, voxel, shape)
            res[name] = {}
            for m in METHODS:
                path = r["paths"].get(m)
                path = [tuple(c) for c in path] if path else None
                res[name][m] = classify2(scene, path, st[m], voxel, objs, pas)
            if ev == REFERENCE:
                # variants: other start/goal pairs and the bypass closed
                for key, (s0, e0) in PAIRS.items():
                    start = (int(s0[0] / voxel), int(s0[1] / voxel))
                    exits = cells_near(e0, voxel, 0.4, shape)
                    for closed in (False, True):
                        if closed and scene not in BARRIER:
                            continue
                        tag = key + ("_closed" if closed else "")
                        out["variants"].setdefault(tag, {}).setdefault(name, {})
                        for m in METHODS:
                            s = st[m].copy()
                            if closed:
                                xs = (np.arange(shape[0]) + 0.5) * voxel
                                ys = (np.arange(shape[1]) + 0.5) * voxel
                                wall = (xs[:, None] >= WALL_X[0]) & (xs[:, None] <= WALL_X[1]) & (ys[None, :] < BARRIER[scene][0])
                                s[wall] = BLOCKED
                            kappa = 0.0 if m == "М3" else 6.0
                            path = plan_route(s, layers["support"], layers["unknown_crawl"], start, exits, voxel, kappa=kappa)
                            out["variants"][tag][name][m] = classify2(scene, path, s, voxel, objs, pas)
            print(ev, name, {m: res[name][m]["class"][0] for m in METHODS}, flush=True)
        out["evals"][ev] = res
    # summaries
    summ = out["summary"]
    for ev, res in out["evals"].items():
        summ[ev] = {m: dict(Counter(v[m]["class"] for v in res.values())) for m in METHODS}
    ref = out["evals"].get(REFERENCE, {})
    per = {"scene": defaultdict(lambda: defaultdict(Counter)), "drone": defaultdict(lambda: defaultdict(Counter))}
    for name, v in ref.items():
        scene, drone = name.split("_")[0], name.split("_")[1]
        for m in METHODS:
            per["scene"][m][scene][v[m]["class"]] += 1
            per["drone"][m][drone][v[m]["class"]] += 1
    summ["per_scene"] = {m: {s: dict(c) for s, c in d.items()} for m, d in per["scene"].items()}
    summ["per_drone"] = {m: {s: dict(c) for s, c in d.items()} for m, d in per["drone"].items()}
    n_ref = len(ref)
    summ["wilson"] = {m: {"safe": wilson(summ[REFERENCE][m].get("safe", 0), n_ref),
                          "unsafe": wilson(summ[REFERENCE][m].get("unsafe", 0), n_ref)} for m in METHODS} if ref else {}
    summ["reasons"] = {ev: {m: [[n, v[m].get("reason")] for n, v in res.items() if v[m]["class"] == "unsafe"] for m in METHODS}
                       for ev, res in out["evals"].items()}
    summ["via"] = {m: dict(Counter(v[m].get("via", "none") for v in ref.values())) for m in METHODS}
    summ["crossings"] = {m: [[n, v[m]["crossing"]] for n, v in ref.items() if v[m].get("crossing")] for m in METHODS}
    summ["variants"] = {tag: {m: dict(Counter(v[m]["class"] for v in d.values())) for m in METHODS}
                        for tag, d in out["variants"].items()}
    summ["variants_via"] = {tag: {m: dict(Counter(v[m].get("via", "none") for v in d.values())) for m in METHODS}
                            for tag, d in out["variants"].items()}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
