"""Зведення оцінки за всіма обльотами: стани проходів і безпечність маршрутів за еталоном.

  python aggregate_eval.py <тека eval> --gt ../data/field_20261104/ground_truth

Безпечність маршруту -- по всіх клітинках шляху (report.json, "paths"): небезпечно, якщо клітинка
лежить в основі предмета без просвіту (стілець, коробка, тринога, стійка тощо; основа звужена на 0,05 м
як допуск вимірювань) або у вимірюваній щілині, вужчій за 0,5 м. Під столом і жердиною (є просвіт)
проходити можна. Безпечний маршрут додатково позначається тим, де він перетинає лінію бар'єра.
"""
import argparse
import csv
import json
import os
import numpy as np
from collections import Counter, defaultdict

BARRIER = {"S2": (2.68, 5.53), "S3": (3.11, 5.53), "S4": (2.68, 5.54), "S5": (2.68, 5.54)}   # y бар'єра
METHODS = ("М3", "М4", "М5", "М5р", "М6")
SHRINK = 0.05


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def obstacles(scene, gt):
    out = []
    for o in read_csv(os.path.join(gt, f"{scene}_objects.csv")):
        x0, y0, x1, y1 = (num(o.get(k, "")) for k in ("x0", "y0", "x1", "y1"))
        if None in (x0, y0, x1, y1) or num(o.get("clearance_m", "")) is not None:
            continue                                      # без координат або з просвітом (стіл, жердина)
        out.append((o["object_id"], min(x0, x1) + SHRINK, min(y0, y1) + SHRINK,
                    max(x0, x1) - SHRINK, max(y0, y1) - SHRINK))
    return out


def classify(scene, path, voxel, gt, passages, cross_y):
    if not path:
        return "немає маршруту"
    obs = obstacles(scene, gt)
    narrow = []
    for p in passages:
        w = num(p.get("width_m", ""))
        if w is not None and w < 0.5 and p.get("y0") and p.get("y1"):
            a, b = sorted((num(p["y0"]), num(p["y1"])))
            narrow.append((p["passage_id"], num(p["x0"]) - 0.3, a, num(p["x0"]) + 0.3, b))
    for i, j in path:
        x, y = (i + 0.5) * voxel, (j + 0.5) * voxel
        for oid, x0, y0, x1, y1 in obs:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return f"НЕБЕЗПЕЧНО: крізь {oid}"
        for pid, x0, y0, x1, y1 in narrow:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return f"НЕБЕЗПЕЧНО: щілина {pid}"
    if scene not in BARRIER:
        return "вільно (без бар'єра)"
    if cross_y is not None and cross_y < BARRIER[scene][0]:
        return "обхід"
    for p in passages:
        if p.get("y0") and p.get("y1") and p["passage_id"] != "BYPASS":
            a, b = sorted((num(p["y0"]), num(p["y1"])))
            if cross_y is not None and a - 0.1 <= cross_y <= b + 0.1:
                return f"прохід {p['passage_id']}"
    return "інше безпечне"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    rows = read_csv(os.path.join(a.eval_dir, "passages.csv"))
    print(f"== {a.eval_dir}: стани проходів (мода в смузі ±0,3 м) проти очікуваного")
    for m in METHODS:
        if m + "_мода" not in rows[0]:
            continue
        hit = sum(r[m + "_мода"] == r["expected"] for r in rows)
        by = defaultdict(Counter)
        for r in rows:
            by[r["expected"]][r[m + "_мода"]] += 1
        print(f"  {m}: збіг {hit} з {len(rows)}; " + "; ".join(f"очікувано «{k}» -> {dict(v)}" for k, v in by.items()))
    rep = json.load(open(os.path.join(a.eval_dir, "report.json"), encoding="utf-8"))
    print("== маршрути від (3,8; 4,0) до (0,6; 2,4)")
    total = defaultdict(Counter)
    for r in rep:
        pas = read_csv(os.path.join(a.gt, f"{r['scene']}_passages.csv"))
        cells = []
        for m in METHODS:
            if m not in r.get("paths", {}):
                continue
            c = classify(r["scene"], r["paths"][m], r["voxel"], a.gt, pas, r["route_cross_y"][m])
            summ = r["routes"].get(m)
            cells.append(f"{m}: {c}" + (f", {summ['time_s']:.0f} с" if summ else ""))
            total[m]["небезпечно" if c.startswith("НЕБЕЗПЕЧНО") else c] += 1
        print(f"  {r['log'][:22]} кадрів {r['frames_accepted']}/{r['frames_total']}: " + " | ".join(cells))
    mq = [r["map_quality"] for r in rep if r.get("map_quality")]
    if mq:
        rec = [q["object_recall"] for q in mq]
        prec = [q["occupied_precision"] for q in mq if q["occupied_precision"] is not None]
        print(f"== якість карти (сцени з предметами, {len(mq)} обльотів): повнота предметів "
              f"{np.mean(rec):.2f} (медіана {np.median(rec):.2f}), точність зайнятих у зоні {np.mean(prec):.2f}")
    print("== підсумок маршрутів за методами:")
    for m, c in total.items():
        print(f"  {m}: {dict(c)}")


if __name__ == "__main__":
    main()
