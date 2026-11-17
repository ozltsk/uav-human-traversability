"""freeze_config.py -- фіксація налаштувань обробки перед підтверджувальною серією E6.

  python -X utf8 freeze_config.py            записати ../drones2026/E6_frozen.json (якщо ще немає)
  python -X utf8 freeze_config.py --verify   звірити чинні файли з записаними контрольними сумами

Файл фіксації не перезаписується: нова редакція можлива лише з --amend "причина" і лише до того,
як прийде архів E6; вона дописується в історію, попередня лишається.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "drones2026", "E6_frozen.json")

FILES = [
    "ac1_evacmap/app.py", "ac1_evacmap/perception.py", "ac1_evacmap/mono_depth.py",
    "ac1_evacmap/occupancy_map.py", "ac1_evacmap/flight_log.py",
    "ac2_evacroute/map_layers.py", "ac2_evacroute/pose_model.py", "ac2_evacroute/nn_traversability.py",
    "ac2_evacroute/route_astar.py",
    "field_analysis/field_eval.py", "field_analysis/aggregate_eval.py", "field_analysis/route_safety2.py",
    "rmtt_field/export_field.py", "rmtt_field/postprocess.py",
    "runs/ac2_models/travnet.pt", "runs/ac2_models_rmtt/travnet.pt",
]

SETTINGS = {
    "export": {"positions": "imu", "fov_deg": 55.5, "calibration": "calib/*.json за серійним номером дрона"},
    "ac1_buildmap": {"voxel_m": 0.1, "max_eps": 0.012, "інші параметри": "типові build_map у ac1_evacmap/app.py",
                     "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
                     "seg_model": "nvidia/segformer-b0-finetuned-ade-512-512"},
    "poses_H_W_v": {"walk": [1.75, 0.5, 1.2], "crouch": [1.2, 0.5, 0.5], "crawl": [0.5, 0.6, 0.25]},
    "estimators": {
        "М3": {"rule_state_map": {"mu": 0.0, "u_max": 1.0}, "route_kappa": 0.0},
        "М4": {"rule_state_map": {"mu": "safety_margin(eps), mu0=1.0, range_m=2.0; eps=0.03, якщо eps<=0",
                                  "u_max": 0.9}, "route_kappa": 6.0},
        "М5": {"model": "runs/ac2_models/travnet.pt", "theta": 0.8, "route_kappa": 6.0},
        "М5р": {"model": "runs/ac2_models_rmtt/travnet.pt", "theta": 0.8, "route_kappa": 6.0},
    },
    "primary_estimator": "М4",
    "passage_state": {"band_half_m": 0.3, "trim": 0.15, "rule": "мода оглянутих клітинок у смузі"},
    "route": {"planner": "route_astar.plan_route", "goal_radius_m": 0.4,
              "start_goal": "з ground_truth/<сцена>_route.csv, записаних виконавицею до польотів"},
    "route_safety": {"script": "field_analysis/route_safety2.py (критерій з урахуванням пози)",
                     "object_base_shrink_m": 0.05, "min_gap_m": 0.5},
    "primary_outcome": "частка обльотів, де М4 дав маршрут без небезпечних клітинок; для сцен без "
                       "прохідного шляху правильна відповідь -- «маршруту немає»",
    "flight_inclusion": "усі польоти E6 з outcome ok і всіма кадрами зйомки; повтори невдалих -- за журналом; "
                        "жоден придатний політ не вилучається після перегляду результатів",
    "scene_specific": "координати бар'єра, зони оцінки карти й точок start/goal беруться з ground_truth E6; "
                      "це дані сцени, а не налаштування методу",
    "S9_S10": "карти польотів A і B зводяться в одну в координатах кімнати без додаткового підбору параметрів",
    "aruco": "положення за маркерами використовується лише для оцінки похибки числення шляху, "
             "у карту основного результату не йде",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    return {p: sha256(os.path.join(ROOT, p)) for p in FILES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--amend", default=None, help="причина нової редакції (лише до приходу архіву E6)")
    a = ap.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    if a.verify:
        with open(OUT, encoding="utf-8") as f:
            cur = json.load(f)["revisions"][-1]
        bad = [p for p, h in cur["sha256"].items() if sha256(os.path.join(ROOT, p)) != h]
        print(f"редакція {cur['revision']} від {cur['frozen_at']}: "
              + ("усі файли збігаються" if not bad else "ЗМІНЕНО: " + ", ".join(bad)))
        sys.exit(1 if bad else 0)
    doc = {"purpose": "Фіксація налаштувань обробки перед підтверджувальною серією E6 (MDPI Drones)",
           "revisions": []}
    if os.path.exists(OUT):
        if not a.amend:
            sys.exit("Файл фіксації вже є. Нова редакція -- лише з --amend \"причина\".")
        with open(OUT, encoding="utf-8") as f:
            doc = json.load(f)
    doc["revisions"].append({"revision": len(doc["revisions"]) + 1, "frozen_at": now,
                             "reason": a.amend or "перша фіксація", "settings": SETTINGS, "sha256": snapshot()})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"записано {OUT}, редакція {len(doc['revisions'])}, файлів {len(FILES)}")


if __name__ == "__main__":
    main()
