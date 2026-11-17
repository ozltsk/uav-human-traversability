"""Запізнення відео щодо телеметрії: взаємна кореляція «руху картинки» (різниця сусідніх кадрів) і
«руху дрона» (|vgz| + |швидкість рискання| + |vgx|+|vgy| з state.csv).

  python video_lag_xcorr.py <тека запису> [<тека запису> ...]
Додатне запізнення -- кадр показує те, що було на lag секунд раніше за його мітку часу.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rmtt_field"))
import postprocess as pp   # noqa: E402


def motion_signals(run, dt=0.1):
    ft = pp.frame_times(run)
    times, diff, prev = [], [], None
    for _i, t, frame in pp.iter_frames(run, ft):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float32)
        if prev is not None:
            times.append(t)
            diff.append(float(np.mean(np.abs(g - prev))))
        prev = g
    times, diff = np.asarray(times), np.asarray(diff)
    st = pp.load_state(run)
    yaw_rate = np.abs(np.gradient(st["yaw_unwrapped"], st["t"]))
    tel = np.abs(st["vgz"]) * 3 + np.hypot(st["vgx"], st["vgy"]) * 2 + yaw_rate / 10.0
    grid = np.arange(max(times[0], st["t"][0]), min(times[-1], st["t"][-1]), dt)
    a = np.interp(grid, times, diff)
    b = np.interp(grid, st["t"], np.nan_to_num(tel))
    return grid, (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)


if __name__ == "__main__":
    for run in sys.argv[1:]:
        grid, a, b = motion_signals(run)
        lags = np.arange(-50, 51)                      # -5..+5 с
        corr = [np.corrcoef(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) - max(0, k)])[0, 1] for k in lags]
        k = int(lags[int(np.argmax(corr))])
        print(f"{os.path.basename(run)}: запізнення відео {k * 0.1:+.1f} с (кореляція {max(corr):.2f}; "
              f"при 0 с -- {corr[50]:.2f})")
