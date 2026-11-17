"""Пакування результатів дослідів для передачі (один zip).

  python pack_results.py                  -> results_<дата>.zip у цій теці
  python pack_results.py --with-mp4       -> разом із video.mp4 (удвічі більший архів)

У архів ідуть: runs/ (усі записи; video.mp4 -- лише з --with-mp4, бо його можна
відновити з video.h264), calib/, ground_truth/ (зокрема фото), missions/, journal.csv,
fleet.json, test_log.txt, CHANGES.md, а також results_summary.csv -- по рядку на кожен запис.
Записи без decode_report.json перед пакуванням декодуються.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postprocess as pp   # noqa: E402

SKIP_NAMES = {"_clean_tmp.h264"}


def _json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def summarize(run_dir):
    meta = _json(os.path.join(run_dir, "meta.json"), {})
    mission = _json(os.path.join(run_dir, "mission.json"), {})
    marks = _json(os.path.join(run_dir, "marks.json"), [])
    dec = _json(os.path.join(run_dir, "decode_report.json"), {})
    drone = meta.get("drone", {}) or {}
    caps = sum(1 for m in marks if m.get("kind") == "capture")
    want = len(mission.get("waypoints", [])) * int(mission.get("turns", 4)) if mission.get("type") == "survey" else ""
    notes = 0
    cpath = os.path.join(run_dir, "commands.csv")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            notes = sum(1 for row in f if ",# примітка," in row)
    dur = (meta.get("t_end", 0) - meta.get("t_start", 0)) if meta.get("t_end") else ""
    return {"run": os.path.basename(run_dir), "started": meta.get("started"), "drone": drone.get("label"),
            "sn": (drone.get("info") or {}).get("sn") or drone.get("sn"), "mission": meta.get("mission"),
            "type": meta.get("type"), "scene": mission.get("scene", ""), "outcome": meta.get("outcome"),
            "duration_s": round(dur, 1) if dur != "" else "", "frames": dec.get("frames"), "fps": dec.get("fps"),
            "max_gap_s": dec.get("max_gap_s"), "state_hz": dec.get("state_rate_hz"), "captures": caps,
            "captures_expected": want, "wifi_recoveries": notes,
            "yaw_sign_flight": (meta.get("result") or {}).get("yaw_sign", ""),
            "blocked": any(m.get("kind") == "blocked" or m.get("blocked") for m in marks)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-mp4", action="store_true")
    ap.add_argument("--no-decode", action="store_true")
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(HERE, "runs", "*")) if os.path.isdir(d))
    rows = []
    for d in runs:
        if not a.no_decode and not os.path.exists(os.path.join(d, "decode_report.json")):
            pp.decode_run(d, write_mp4=False, verbose=False)
        rows.append(summarize(d))
    summary = os.path.join(HERE, "results_summary.csv")
    with open(summary, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        w.writeheader()
        w.writerows(rows)
    out = a.out or os.path.join(HERE, f"results_{time.strftime('%Y%m%d_%H%M')}.zip")
    n_files = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        def add(path):
            nonlocal n_files
            arc = os.path.relpath(path, HERE)
            z.write(path, arc)
            n_files += 1

        for top in ("runs", "calib", "ground_truth", "missions"):
            for root, _dirs, files in os.walk(os.path.join(HERE, top)):
                for name in files:
                    if name in SKIP_NAMES or (name == "video.mp4" and not a.with_mp4):
                        continue
                    add(os.path.join(root, name))
        for name in ("journal.csv", "fleet.json", "test_log.txt", "CHANGES.md", "results_summary.csv"):
            if os.path.exists(os.path.join(HERE, name)):
                add(os.path.join(HERE, name))
    size = os.path.getsize(out) / 1e6
    print(f"Записів: {len(rows)}; файлів в архіві: {n_files}; {out} ({size:.0f} МБ)")
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["outcome"] == "ok")
    for t, oks in sorted(by_type.items(), key=lambda kv: str(kv[0])):
        print(f"  {t}: {len(oks)} записів, успішних {sum(oks)}")
    if not os.path.exists(os.path.join(HERE, "journal.csv")):
        print("УВАГА: journal.csv не знайдено -- без журналу виміряних точок посадки результати неповні")


if __name__ == "__main__":
    main()
