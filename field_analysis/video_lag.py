"""Перевірка часу кадрів: кадри навколо подій (ToF-покази, посадка) у записі E4b зі стіною.

  python video_lag.py <тека запису> <вихідний png>
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rmtt_field"))
import postprocess as pp   # noqa: E402

run, out = sys.argv[1], sys.argv[2]
ft = pp.frame_times(run)
times = ft["times"]
tofs = [(float(r["t"]), int(r["tof_mm"])) for r in csv.DictReader(open(os.path.join(run, "ext_tof.csv"), encoding="utf-8")) if r["tof_mm"]]
cmds = list(csv.reader(open(os.path.join(run, "commands.csv"), encoding="utf-8")))[1:]
land = next(float(r[0]) for r in cmds if r[2] == "land")
st = pp.load_state(run)
events = [(t, f"ToF {mm} мм") for t, mm in tofs if 150 < mm < 1300] + [(land, "land"), (land + 2.0, "land+2с"), (land + 4.0, "land+4с")]
want = {}
for t, label in events:
    i = int(np.argmin(np.abs(times - t)))
    h = float(pp.interp_state(st, "tof", t))
    want[i] = f"{label}, t={t - times[0]:.1f} с, кадр {i}, h {h:.0f} см"
tiles = []
for i, _t, frame in pp.iter_frames(run, ft):
    if i in want:
        img = cv2.resize(frame, (480, 360))
        cv2.rectangle(img, (0, 0), (480, 22), (0, 0, 0), -1)
        cv2.putText(img, want[i], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
cv2.imwrite(out, grid)
print("land at", round(land - times[0], 2), "с; кадрів", len(times), "; останній кадр", round(times[-1] - times[0], 2), "с")
print(want)
