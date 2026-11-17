# -*- coding: utf-8 -*-
"""tape_map.py -- пункт 3 зауважень: карта з даних рулетки замість карти дрона (16.11.2026).

Для кожної сцени першої серії будується точна сітка зайнятості з виміряних предметів (основа, висота, просвіт),
без невідомих вокселів, і на ній запускаються ті самі оцінювачі (М3, М4, М5, М5р), той самий A* і та сама
перевірка маршруту. Якщо на такій карті стани проходів і маршрути правильні, помилки натурного досліду
походять від карти дрона, а не від оцінювача.

Припущення, де висоту не міряли: предмет без просвіту -- 1,0 м (стільці 0,8 м, стійки вищі; на стан колонки
не впливає, бо будь-яка висота понад 0 блокує), предмет із просвітом -- плита 0,1 м над просвітом.
Парти вздовж дальньої стіни (y > 5,25 м) -- зайняті до 0,75 м.

    python -X utf8 tape_map.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac2_evacroute"))
sys.path.insert(0, HERE)
import route_safety2 as rs2                                             # noqa: E402
from field_eval import read_csv, num, expected_state, STATE_UA          # noqa: E402
from map_layers import column_layers, features_from_layers             # noqa: E402
from pose_model import rule_state_map, safety_margin, UNSEEN            # noqa: E402
from nn_traversability import load_model, predict_probs, decide         # noqa: E402
from route_astar import plan_route, cells_near                          # noqa: E402

VOXEL, ROOM, NZ, EPS, THETA = 0.1, (5.0, 6.0), 30, 0.006, 0.8
SCENES = ("S1", "S2", "S3", "S4", "S5")
EN = {0: "walk", 1: "crouch", 2: "crawl", 3: "blocked", 4: "unseen"}


def truth_grid(scene):
    nx, ny = int(round(ROOM[0] / VOXEL)), int(round(ROOM[1] / VOXEL))
    occ = np.zeros((nx, ny, NZ), dtype=bool)
    xs, ys = (np.arange(nx) + 0.5) * VOXEL, (np.arange(ny) + 0.5) * VOXEL
    for o in read_csv(os.path.join(rs2.GT, f"{scene}_objects.csv")):
        v = [num(o.get(k, "")) for k in ("x0", "y0", "x1", "y1")]
        if None in v:
            continue
        x0, x1, y0, y1 = min(v[0], v[2]), max(v[0], v[2]), min(v[1], v[3]), max(v[1], v[3])
        m = ((xs[:, None] >= x0) & (xs[:, None] <= x1)) & ((ys[None, :] >= y0) & (ys[None, :] <= y1))
        h, c = num(o.get("height_m", "")), num(o.get("clearance_m", ""))
        k0 = 0 if c is None else int(round(c / VOXEL))
        k1 = int(round((h if h is not None else (1.0 if c is None else c + 0.1)) / VOXEL))
        occ[m, k0:max(k1, k0 + 1)] = True
    occ[:, ys > rs2.DESKS_Y, :8] = True
    return occ


def main():
    model = load_model(os.path.join(ROOT, "runs", "ac2_models", "travnet.pt"), "cpu")
    model_r = load_model(os.path.join(ROOT, "runs", "ac2_models_rmtt", "travnet.pt"), "cpu")
    out, agree, routes = {}, Counter(), {}
    for scene in SCENES:
        occ = truth_grid(scene)
        layers = column_layers(occ.astype(np.int8) * 2, VOXEL)
        feats = features_from_layers(layers)
        st = {"М3": rule_state_map(layers, VOXEL, mu=0.0, u_max=1.0),
              "М4": rule_state_map(layers, VOXEL, mu=safety_margin(EPS), u_max=0.9),
              "М5": decide(predict_probs(model, feats, "cpu"), theta=THETA, unknown_crawl=layers["unknown_crawl"]),
              "М5р": decide(predict_probs(model_r, feats, "cpu"), theta=THETA, unknown_crawl=layers["unknown_crawl"])}
        shape = st["М3"].shape
        objs, pas = rs2.gt_objects(scene), rs2.gt_passages(scene, VOXEL, shape)
        row = {"passages": {}, "routes": {}}
        for p in pas:
            exp = expected_state(p["width"], p["clearance"])
            row["passages"][p["id"]] = {"expected": EN[exp]}
            for m, s in st.items():
                vals = np.array([s[c] for c in sorted(p["cells"])])
                seen = vals[vals != UNSEEN]
                best = int(np.bincount(seen, minlength=5).argmax()) if len(seen) else UNSEEN
                fastest = int(vals.min())
                row["passages"][p["id"]][m] = {"modal": EN[best], "fastest_cell": EN[fastest]}
                agree[(m, "modal", best == exp)] += 1
                agree[(m, "fastest", fastest == exp)] += 1
        for key, (s0, e0) in rs2.PAIRS.items():
            start = (int(s0[0] / VOXEL), int(s0[1] / VOXEL))
            exits = cells_near(e0, VOXEL, 0.4, shape)
            for m, s in st.items():
                path = plan_route(s, layers["support"], layers["unknown_crawl"], start, exits, VOXEL,
                                  kappa=0.0 if m == "М3" else 6.0)
                c = rs2.classify2(scene, path, s, VOXEL, objs, pas)
                row["routes"].setdefault(key, {})[m] = {k: c.get(k) for k in ("class", "reason", "via", "crossing")}
                routes.setdefault(key, {}).setdefault(m, Counter())[c["class"]] += 1
        row["share_walk"] = {m: round(float((s == 0).mean()), 3) for m, s in st.items()}
        out[scene] = row
        print(scene, json.dumps(row["passages"], ensure_ascii=False))
        print("   маршрути A:", {m: (v["class"], v["via"]) for m, v in row["routes"]["A"].items()}, row["share_walk"])
    n = sum(len(out[s]["passages"]) for s in SCENES)
    summ = {"passages_n": n,
            "agree_modal": {m: agree[(m, "modal", True)] for m in ("М3", "М4", "М5", "М5р")},
            "agree_fastest_cell": {m: agree[(m, "fastest", True)] for m in ("М3", "М4", "М5", "М5р")},
            "routes": {k: {m: dict(c) for m, c in d.items()} for k, d in routes.items()}}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    json.dump({"summary": summ, "scenes": out}, open(os.path.join(rs2.RUNS, "tape_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
