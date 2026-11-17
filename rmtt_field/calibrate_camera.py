"""Калібрування камери дрона за відео шахівниці 9 x 6 внутрішніх кутів, клітинка 25 мм.

Запис (дрон у руці, без зльоту): python app.py bench --ap --label 3 --seconds 45 --name calib
Під час запису повільно водити дрон перед шахівницею (checkerboard_A4.pdf, друк 100 %,
наклеєна на рівний картон): різні нахили до 45 градусів, відстань 0,4-1,2 м, шахівниця
в центрі й по кутах кадру. Потім:
  python calibrate_camera.py runs/<тека запису> --out calib/calib_d3.json
Приймає також відеофайл або теку із зображеннями. Добре калібрування: rms < 1 пікс., ракурсів >= 15.
Формат calib.json: K, dist, width, height (його читає postprocess.export_ac1 через camera.calib).
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postprocess as pp   # noqa: E402


def frames_from(src, step):
    if os.path.isdir(src) and os.path.exists(os.path.join(src, "video.h264")):
        for i, _t, frame in pp.iter_frames(src):
            if i % step == 0:
                yield frame
    elif os.path.isdir(src):
        for path in sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg"))):
            yield cv2.imread(path)
    else:
        cap = cv2.VideoCapture(src)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield frame
            i += 1


def calibrate(src, cols=9, rows=6, square_mm=25.0, step=6, max_views=40, min_shift_px=25.0):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm / 1000.0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    obj, img, size, seen, found = [], [], None, 0, 0
    for frame in frames_from(src, step):
        seen += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not ok:
            continue
        found += 1
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        # майже той самий ракурс, що вже взятий, нічого не додає
        if any(np.mean(np.linalg.norm(corners - c, axis=2)) < min_shift_px for c in img):
            continue
        obj.append(objp)
        img.append(corners)
    if len(obj) < 8:
        raise SystemExit(f"Шахівницю знайдено в {found} кадрах із {seen}, різних ракурсів {len(obj)}; потрібно >= 8. "
                         f"Повторіть запис повільніше, при кращому світлі, з різних кутів.")
    if len(obj) > max_views:
        idx = np.linspace(0, len(obj) - 1, max_views).astype(int)
        obj, img = [obj[i] for i in idx], [img[i] for i in idx]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "width": int(size[0]), "height": int(size[1]),
            "rms_px": round(float(rms), 3), "views": len(obj), "frames_with_board": found, "frames_seen": seen,
            "hfov_deg": round(math.degrees(2 * math.atan(size[0] / (2 * K[0, 0]))), 2),
            "board": {"cols": cols, "rows": rows, "square_mm": square_mm}, "source": os.path.abspath(src)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="тека запису, відеофайл або тека зображень")
    ap.add_argument("--out", required=True)
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--step", type=int, default=6, help="брати кожен N-й кадр")
    a = ap.parse_args()
    cal = calibrate(a.src, square_mm=a.square_mm, step=a.step)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    ok = cal["rms_px"] < 1.0 and cal["views"] >= 15
    print(f"rms {cal['rms_px']} пікс., ракурсів {cal['views']}, кадрів із шахівницею {cal['frames_with_board']} "
          f"з {cal['frames_seen']}, fx {cal['K'][0][0]:.1f}, кут огляду {cal['hfov_deg']}° -> {a.out}")
    print("Якість: " + ("добра" if ok else "НЕДОСТАТНЯ (потрібно rms < 1 і ракурсів >= 15) -- повторіть запис"))


if __name__ == "__main__":
    main()
