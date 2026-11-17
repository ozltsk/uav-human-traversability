# -*- coding: utf-8 -*-
"""scan_people.py -- перед публікацією: шукає людей у відео дронів (1 кадр за секунду, YOLO11n, клас person).

    python -X utf8 scan_people.py <тека runs> <ваги .pt> <тека звіту>

Пише <звіт>/people_scan.csv (запис, секунд, кадрів перевірено, кадрів із людиною, найбільша впевненість, найбільша
частка кадру) і для кожного запису з виявленням -- зменшений кадр із рамкою. Нічого не змінює у вихідних даних.
"""
import csv
import os
import sys

import av
from ultralytics import YOLO

runs, weights, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
model = YOLO(weights)
rows = []
for name in sorted(os.listdir(runs)):
    path = os.path.join(runs, name, "video.h264")
    if not os.path.exists(path):
        continue
    n = hits = 0
    best = (0.0, 0.0, None)
    try:
        with av.open(path) as c:
            for i, frame in enumerate(c.decode(video=0)):
                if i % 24:
                    continue
                img = frame.to_ndarray(format="bgr24")
                r = model.predict(img, imgsz=480, conf=0.30, classes=[0], verbose=False)[0]
                n += 1
                if len(r.boxes):
                    hits += 1
                    conf = float(r.boxes.conf.max())
                    k = int(r.boxes.conf.argmax())
                    x0, y0, x1, y1 = r.boxes.xyxy[k].tolist()
                    area = (x1 - x0) * (y1 - y0) / (img.shape[0] * img.shape[1])
                    if conf > best[0]:
                        best = (conf, area, r.plot())
    except Exception as e:                                   # обірваний потік наприкінці запису -- не помилка
        print(name, "decode stopped:", str(e)[:80], flush=True)
    if best[2] is not None:
        import cv2
        cv2.imwrite(os.path.join(out, name + ".jpg"), cv2.resize(best[2], (480, 360)))
    rows.append([name, n, hits, round(best[0], 2), round(best[1], 3)])
    print(*rows[-1], flush=True)
with open(os.path.join(out, "people_scan.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run", "frames_checked", "frames_with_person", "max_conf", "max_area_share"])
    w.writerows(rows)
print("done")
