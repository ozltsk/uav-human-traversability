"""Карта лише з кадрів однієї точки облету (усі кадри мають одне положення -- похибки положення між
точками немає) і стани колонок у смузі проходу.

  python single_waypoint.py <лог АС Р1> <номер точки з 0> <x0,y0,x1,y1 проходу> [--out <тека>]
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac2_evacroute"))
from map_layers import load_map_npz, column_layers   # noqa: E402
from pose_model import rule_state_map                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("wp", type=int)
ap.add_argument("seg")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or a.log.rstrip("/\\") + f"_wp{a.wp}"
os.makedirs(os.path.join(out, "frames"), exist_ok=True)
os.makedirs(os.path.join(out, "pred"), exist_ok=True)
shutil.copy(os.path.join(a.log, "meta.json"), out)
rows = open(os.path.join(a.log, "telemetry.csv"), encoding="utf-8").read().splitlines()
keep = [rows[0]] + [r for r in rows[1:] if int(r.split(",")[5]) == a.wp]
for r in keep[1:]:
    i = int(r.split(",")[0])
    for sub, ext in (("frames", "png"), ("pred", "npz")):
        name = f"{'frame' if sub == 'frames' else 'pred'}_{i:04d}.{ext}"
        if os.path.exists(os.path.join(a.log, sub, name)):
            shutil.copy(os.path.join(a.log, sub, name), os.path.join(out, sub, name))
open(os.path.join(out, "telemetry.csv"), "w", encoding="utf-8").write("\n".join(keep) + "\n")
npz = os.path.join(out, "map.npz")
subprocess.run([sys.executable, os.path.join(ROOT, "ac1_evacmap", "app.py"), "buildmap", "--log", out, "--out", npz],
               check=True, stdout=subprocess.DEVNULL, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
state, voxel, eps = load_map_npz(npz)
layers = column_layers(state, voxel)
st = rule_state_map(layers, voxel, mu=0.0, u_max=1.0)
x0, y0, x1, y1 = (float(v) for v in a.seg.split(","))
names = ("пішки", "пригн.", "повзком", "непрох.", "не огл.")
print(f"точка {a.wp + 1}, кадрів {len(keep) - 1}; смуга проходу ({x0},{y0})-({x1},{y1}):")
for x in np.arange(x0 - 0.3, x1 + 0.31, 0.1):
    line = []
    for y in np.arange(min(y0, y1) - 0.3, max(y0, y1) + 0.31, 0.1):
        i, j = int(x / voxel), int(y / voxel)
        occ = (state[i, j, 3:19] == 2).sum()
        line.append(f"{names[st[i, j]][:4]}{'' if occ == 0 else '#' + str(occ)}")
    print(f"  x {x:.2f}: " + " ".join(f"{c:9}" for c in line))
print("   (y від", round(min(y0, y1) - 0.3, 2), "до", round(max(y0, y1) + 0.3, 2), "через 0,1 м; #N -- зайнятих вокселів на 0,3-1,9 м)")
