"""Діагностика колонок карти: профіль станів по висоті та шари АС Р2 у вибраних точках."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ac2_evacroute"))
from map_layers import load_map_npz, column_layers   # noqa: E402

npz = sys.argv[1]
pts = [tuple(map(float, p.split(","))) for p in sys.argv[2:]]
state, voxel, eps = load_map_npz(npz)
layers = column_layers(state, voxel)
print("форма", state.shape, "eps", round(eps, 3), "шари:", sorted(layers))
sym = {0: ".", 1: "?", 2: "#"}
for x, y in pts:
    i, j = int(x / voxel), int(y / voxel)
    col = state[i, j, :]
    prof = "".join(sym[int(v)] for v in col)
    info = {k: round(float(layers[k][i, j]), 2) for k in ("support", "clearance", "unknown_walk", "unknown_crawl",
                                                           "observed") if k in layers}
    print(f"({x:.1f}, {y:.1f}) z 0..{col.size * voxel:.1f} м: {prof}  {info}")
# загальна картина по висоті
occ_by_z = (state == 2).mean(axis=(0, 1))
free_by_z = (state == 0).mean(axis=(0, 1))
print("частка зайнятих по z (кожні 0,1 м):", " ".join(f"{v:.2f}" for v in occ_by_z[:25]))
print("частка вільних по z:            ", " ".join(f"{v:.2f}" for v in free_by_z[:25]))
