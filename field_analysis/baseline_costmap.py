# -*- coding: utf-8 -*-
"""baseline_costmap.py -- відомий базовий метод для порівняння (пункт 5 зауважень, 16.11.2026).

Класична двовимірна сітка зайнятості з розширенням перешкод на радіус тіла (інфляційний шар, як у
costmap_2d навігаційного стека ROS): колонка -- перешкода, якщо в смузі висот [z_min, z_max) є зайнятий
воксел; перешкоди розширюються на половину ширини тіла; решта клітинок прохідні пішки. Пози, просвіт
під предметами і невизначеність не враховуються. Два трактування неоглянутих колонок: вільні (типове
для costmap_2d із track_unknown_space=false) і перешкода.

Ті самі карти дрона (eval18_eps012_th08), той самий A* (kappa 0), та сама перевірка route_safety2.classify2.
Жодного параметра не підібрано за результатом: z_min 0,1 м, z_max 1,75 м (зріст), радіус 0,25 м.

    python -X utf8 baseline_costmap.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac2_evacroute"))
sys.path.insert(0, HERE)
import route_safety2 as rs2                                             # noqa: E402
from map_layers import load_map_npz, OCCUPIED, UNKNOWN                  # noqa: E402
from pose_model import WALK, BLOCKED                                    # noqa: E402
from route_astar import plan_route, cells_near                          # noqa: E402
from aggregate_eval import BARRIER                                      # noqa: E402

Z_MIN, Z_MAX, RADIUS = 0.1, 1.75, 0.25
EV = os.path.join(rs2.RUNS, rs2.REFERENCE)


def costmap_state(npz, unknown_blocked):
    state3d, voxel, _ = load_map_npz(npz)
    k0, k1 = int(round(Z_MIN / voxel)), int(round(Z_MAX / voxel))
    band = state3d[:, :, k0:k1]
    obst = (band == OCCUPIED).any(axis=2)
    if unknown_blocked:
        obst |= (band == UNKNOWN).all(axis=2)
    free = distance_transform_edt(~obst) * voxel > RADIUS
    return np.where(free, WALK, BLOCKED).astype(np.int8), voxel


def main():
    rep = json.load(open(os.path.join(EV, "report.json"), encoding="utf-8"))
    out = {}
    for r in rep:
        name, scene = r["log"], r["scene"]
        for ub in (False, True):
            st, voxel = costmap_state(os.path.join(EV, name + ".npz"), ub)
            shape = st.shape
            zeros = np.zeros(shape, dtype=float)
            objs, pas = rs2.gt_objects(scene), rs2.gt_passages(scene, voxel, shape)
            for key, (s0, e0) in rs2.PAIRS.items():
                start = (int(s0[0] / voxel), int(s0[1] / voxel))
                exits = cells_near(e0, voxel, 0.4, shape)
                for closed in (False, True):
                    if closed and scene not in BARRIER:
                        continue
                    s = st.copy()
                    if closed:
                        xs = (np.arange(shape[0]) + 0.5) * voxel
                        ys = (np.arange(shape[1]) + 0.5) * voxel
                        s[(xs[:, None] >= rs2.WALL_X[0]) & (xs[:, None] <= rs2.WALL_X[1])
                          & (ys[None, :] < BARRIER[scene][0])] = BLOCKED
                    path = plan_route(s, zeros, zeros, start, exits, voxel, kappa=0.0)
                    tag = ("unk_blocked" if ub else "unk_free") + "/" + key + ("_closed" if closed else "")
                    out.setdefault(tag, {})[name] = rs2.classify2(scene, path, s, voxel, objs, pas)
    summ = {}
    for tag, d in out.items():
        c = Counter(v["class"] for v in d.values())
        summ[tag] = {"n": len(d), **dict(c),
                     "reasons": dict(Counter(v.get("reason") for v in d.values() if v["class"] == "unsafe")),
                     "via": dict(Counter(v.get("via") for v in d.values() if v["class"] == "safe")),
                     "per_scene_safe": dict(Counter(n.split("_")[0] for n, v in d.items() if v["class"] == "safe"))}
        print(tag, json.dumps(summ[tag], ensure_ascii=False))
    json.dump({"settings": {"z_min": Z_MIN, "z_max": Z_MAX, "radius": RADIUS}, "summary": summ, "runs": out},
              open(os.path.join(rs2.RUNS, "baseline_costmap.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
