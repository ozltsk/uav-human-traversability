"""Таблиця абляцій: кілька тек оцінки (field_eval) в одній таблиці.

  python ablation_table.py --gt ../data/field_20261104/ground_truth назва=тека [назва=тека ...] [--out table.md]

Для кожної теки: обльотів; якість карти (повнота предметів, точність зайнятих у зоні); збіг станів
проходів (із 23 або з підмножини); маршрути за методами: знайдено / небезпечних / немає.
Безпечність -- як в aggregate_eval (по всіх клітинках маршруту).
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate_eval import classify, read_csv, METHODS   # noqa: E402


def summarize(eval_dir, gt):
    rep = json.load(open(os.path.join(eval_dir, "report.json"), encoding="utf-8"))
    rows = read_csv(os.path.join(eval_dir, "passages.csv"))
    out = {"n": len(rep)}
    mq = [r["map_quality"] for r in rep if r.get("map_quality")]
    out["recall"] = float(np.mean([q["object_recall"] for q in mq])) if mq else None
    pr = [q["occupied_precision"] for q in mq if q and q["occupied_precision"] is not None]
    out["precision"] = float(np.mean(pr)) if pr else None
    out["passages"] = {m: (sum(r[m + "_мода"] == r["expected"] for r in rows), len(rows))
                       for m in METHODS if rows and m + "_мода" in rows[0]}
    routes = {}
    for m in METHODS:
        c = Counter()
        for r in rep:
            if m not in r.get("paths", {}):
                continue
            pas = read_csv(os.path.join(gt, f"{r['scene']}_passages.csv"))
            cls = classify(r["scene"], r["paths"][m], r["voxel"], gt, pas, r["route_cross_y"][m])
            c["unsafe" if cls.startswith("НЕБЕЗПЕЧНО") else ("none" if cls == "немає маршруту" else "safe")] += 1
        if sum(c.values()):
            routes[m] = c
    out["routes"] = routes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="назва=тека")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = ["| варіант | обльотів | повнота предметів | точність зайнятих | проходи: збіг | " +
             " | ".join(f"{m}: безпечних / небезпечних / немає" for m in METHODS) + " |",
             "|---|---|---|---|---|" + "---|" * len(METHODS)]
    for item in a.items:
        name, d = item.split("=", 1)
        s = summarize(d, a.gt)
        pas = ", ".join(f"{m} {h}/{n}" for m, (h, n) in s["passages"].items())
        cells = []
        for m in METHODS:
            c = s["routes"].get(m)
            cells.append("—" if not c else f"{c['safe']} / {c['unsafe']} / {c['none']}")
        lines.append(f"| {name} | {s['n']} | {s['recall']:.2f} | {s['precision']:.2f} | {pas} | " + " | ".join(cells) + " |"
                     if s["recall"] is not None else f"| {name} | {s['n']} | — | — | {pas} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
