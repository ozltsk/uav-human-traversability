"""Обробка запису після посадки.

decode_run  -- сирий H.264 + індекс пакетів -> час кожного кадру, video.mp4,
               frames.csv (час, різкість, телеметрія на момент кадру), звіт;
export_ac1  -- кадри в точках зависання -> журнал обльоту АС Р1
               (meta.json, telemetry.csv, frames/), далі `ac1_evacmap/app.py predict`
               і `buildmap`.

Час кадру -- момент надходження останнього пакета кадру. Межі кадрів
знаходяться розбором потоку: новий кадр починає зріз (NAL 1 або 5) з
first_mb_in_slice = 0. Потік обрізається до першого SPS, бо без нього
декодер відкидає кадри й нумерація зсувається.
"""
import csv
import glob
import json
import math
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_START = re.compile(b"\x00\x00\x01")
TELEMETRY_KEYS = ["tof", "h", "yaw", "pitch", "roll", "vgx", "vgy", "vgz", "bat", "temph"]


# ---------------------------------------------------------------- розбір потоку
def nal_scan(data):
    """(зсуви SPS, зсуви початків кадрів) у сирому потоці Annex B."""
    sps, pics = [], []
    n = len(data)
    for m in _START.finditer(data):
        i = m.end()
        if i + 1 >= n:
            break
        nal_type = data[i] & 0x1F
        if nal_type == 7:
            sps.append(m.start())
        elif nal_type in (1, 5) and data[i + 1] & 0x80:     # ue(v) first_mb_in_slice == 0
            pics.append(m.start())
    return sps, pics


def _read_index(path):
    t, off = [], []
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                t.append(float(parts[0]))
                off.append(int(parts[1]))
    return np.asarray(t), np.asarray(off, dtype=np.int64)


def frame_times(run_dir):
    """Розбір video.h264: дані для декодування і час кожного кадру."""
    raw_path = os.path.join(run_dir, "video.h264")
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        return None
    with open(raw_path, "rb") as f:
        data = f.read()
    t_pkt, off_pkt = _read_index(os.path.join(run_dir, "video_index.csv"))
    sps, pics = nal_scan(data)
    if not sps:
        return {"data": b"", "times": np.zeros(0), "packets": len(t_pkt), "bytes": len(data),
                "t_first_packet": float(t_pkt[0]) if len(t_pkt) else None}
    start = sps[0]
    kept = [p for p in pics if p >= start]
    ends = kept[1:] + [len(data)]
    end_bytes = np.asarray(ends, dtype=np.int64) - 1
    k = np.searchsorted(off_pkt, end_bytes, side="right") - 1
    times = t_pkt[np.clip(k, 0, len(t_pkt) - 1)]
    return {"data": data[start:], "times": times, "packets": int(len(t_pkt)), "bytes": len(data),
            "t_first_packet": float(t_pkt[0]), "t_last_packet": float(t_pkt[-1])}


def iter_frames(run_dir, ft=None):
    """Генератор (номер, час, кадр BGR). Тимчасовий обрізаний файл видаляється."""
    ft = ft or frame_times(run_dir)
    if ft is None or not ft["data"]:
        return
    tmp = os.path.join(run_dir, "_clean_tmp.h264")
    with open(tmp, "wb") as f:
        f.write(ft["data"])
    cap = cv2.VideoCapture(tmp)
    try:
        i = 0
        times = ft["times"]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = float(times[i]) if i < len(times) else float("nan")
            yield i, t, frame
            i += 1
    finally:
        cap.release()
        os.remove(tmp)


# ---------------------------------------------------------------- телеметрія
def load_state(run_dir):
    path = os.path.join(run_dir, "state.csv")
    cols = {k: [] for k in ["t"] + TELEMETRY_KEYS}
    if not os.path.exists(path):
        return {k: np.zeros(0) for k in cols}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                vals = {k: float(row[k]) if row.get(k) not in (None, "") else np.nan for k in cols}
            except ValueError:
                continue
            for k in cols:
                cols[k].append(vals[k])
    out = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    if len(out["yaw"]):
        good = ~np.isnan(out["yaw"])
        y = out["yaw"].copy()
        y[good] = np.degrees(np.unwrap(np.radians(y[good])))
        out["yaw_unwrapped"] = y
    else:
        out["yaw_unwrapped"] = np.zeros(0)
    return out


def interp_state(st, key, t):
    if len(st["t"]) == 0:
        return np.full(np.shape(t), np.nan)
    v = st[key]
    good = ~np.isnan(v)
    if not good.any():
        return np.full(np.shape(t), np.nan)
    return np.interp(t, st["t"][good], v[good])


def state_window(st, key, t0, t1):
    if len(st["t"]) == 0:
        return np.zeros(0)
    sel = (st["t"] >= t0) & (st["t"] <= t1) & ~np.isnan(st[key])
    return st[key][sel]


# ---------------------------------------------------------------- декодування
def sharpness(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if g.shape[1] > 480:
        g = cv2.resize(g, (480, int(round(g.shape[0] * 480 / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def decode_run(run_dir, write_mp4=True, verbose=True):
    """frames.csv + video.mp4 + first_frame.png + decode_report.json."""
    ft = frame_times(run_dir)
    report = {"run": os.path.basename(os.path.normpath(run_dir))}
    if ft is None:
        report.update({"frames": 0, "note": "відео немає (video.h264 порожній)"})
        _save_json(os.path.join(run_dir, "decode_report.json"), report)
        if verbose:
            print(f"{report['run']}: відео немає")
        return report
    st = load_state(run_dir)
    rows, writer, shape = [], None, None
    times = ft["times"]
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    fps_nominal = (len(times) - 1) / span if span > 0 else 30.0    # RMTT дає ~24 к/с, не 30
    for i, t, frame in iter_frames(run_dir, ft):
        if shape is None:
            shape = frame.shape
            cv2.imwrite(os.path.join(run_dir, "first_frame.png"), frame)
            if write_mp4:
                writer = cv2.VideoWriter(os.path.join(run_dir, "video.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps_nominal,
                                         (frame.shape[1], frame.shape[0]))
        if writer is not None:
            writer.write(frame)
        rows.append((i, t, sharpness(frame)))
    if writer is not None:
        writer.release()
    n_au = len(ft["times"])
    t_arr = np.asarray([r[1] for r in rows], dtype=np.float64)
    tel = {k: interp_state(st, k, t_arr) for k in TELEMETRY_KEYS}
    yaw_u = interp_state(st, "yaw_unwrapped", t_arr) if len(st["t"]) else np.full(len(t_arr), np.nan)
    with open(os.path.join(run_dir, "frames.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "t", "t_rel", "sharpness"] + TELEMETRY_KEYS + ["yaw_unwrapped"])
        t0 = t_arr[0] if len(t_arr) else 0.0
        for j, (i, t, s) in enumerate(rows):
            w.writerow([i, f"{t:.4f}", f"{t - t0:.4f}", f"{s:.1f}"] +
                       [f"{tel[k][j]:.2f}" for k in TELEMETRY_KEYS] + [f"{yaw_u[j]:.2f}"])
    gaps = np.diff(t_arr) if len(t_arr) > 1 else np.zeros(0)
    dur = float(t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 0.0
    report.update({
        "frames": len(rows), "frames_in_stream": n_au, "count_match": len(rows) == n_au,
        "width": int(shape[1]) if shape else None, "height": int(shape[0]) if shape else None,
        "duration_s": round(dur, 3), "fps": round((len(rows) - 1) / dur, 2) if dur > 0 else None,
        "max_gap_s": round(float(gaps.max()), 3) if len(gaps) else None,
        "gaps_over_0_2s": int((gaps > 0.2).sum()) if len(gaps) else 0,
        "packets": ft["packets"], "bytes": ft["bytes"],
        "bitrate_kbps": round(8 * ft["bytes"] / 1000 / max(ft["t_last_packet"] - ft["t_first_packet"], 1e-3), 1)
        if ft.get("t_last_packet") else None,
        "state_samples": int(len(st["t"])),
        "state_rate_hz": round((len(st["t"]) - 1) / (st["t"][-1] - st["t"][0]), 1) if len(st["t"]) > 1 else None,
    })
    _save_json(os.path.join(run_dir, "decode_report.json"), report)
    if verbose:
        print(f"{report['run']}: кадрів {report['frames']} (у потоці {n_au}), {report['width']}x{report['height']}, "
              f"{report['fps']} к/с, тривалість {report['duration_s']} с, найбільша пауза {report['max_gap_s']} с, "
              f"потік {report['bitrate_kbps']} кбіт/с, телеметрія {report['state_rate_hz']} Гц")
        if not report["count_match"]:
            print("  УВАГА: кількість декодованих кадрів не збігається з розбором потоку -- час кадрів наближений")
    return report


# ---------------------------------------------------------------- експорт у АС Р1
def _load_calib(path):
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    return np.asarray(cal["K"], float), np.asarray(cal["dist"], float).ravel()


HFOV_DEFAULT = 55.5    # калібрування 4 дронів RMTT, 02.11.2026: 55,0-55,7° при 960 x 720 (не 70°)


def find_calib(run_dir, calib_dir=None):
    """calib_*.json, у якому sn збігається з серійним номером дрона цього запису (None, якщо немає)."""
    calib_dir = calib_dir or os.path.join(HERE, "calib")
    try:
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            drone = json.load(f).get("drone") or {}
    except (OSError, ValueError):
        return None
    sn = (drone.get("info") or {}).get("sn") or drone.get("sn")
    if not sn:
        return None
    for path in sorted(glob.glob(os.path.join(calib_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                if json.load(f).get("sn") == sn:
                    return path
        except (OSError, ValueError):
            continue
    return None


def forward_track(run_dir, mission, st, yaw_ref, yaw_sign):
    """Положення за виконаними «forward N» з фактичним курсом із рискання: [(t кінця кроку, x, y)].
    Крок вважається виконаним, якщо прийшло «ok» (зокрема із запізненням) або примітка «виконано»."""
    with open(os.path.join(run_dir, "commands.csv"), encoding="utf-8", newline="") as f:
        cmds = list(csv.reader(f))[1:]
    heading = float(mission.get("heading_deg", 0.0))
    pos = np.asarray(mission["origin"], float)
    track = [(0.0, float(pos[0]), float(pos[1]))]
    for i, r in enumerate(cmds):
        if len(r) < 4 or not r[2].startswith("forward"):
            continue
        done = r[3].lower().startswith("ok")
        if r[3] == "TIMEOUT":
            nxt = next((c for c in cmds[i + 1:i + 4] if c[2].startswith("#")), None)
            done = nxt is not None and ("виконано" in nxt[3] or "із запізненням" in nxt[3])
        if not done:
            continue
        t0 = float(r[0])
        t1 = float(r[1]) if r[1] else t0 + 3.0
        yw = state_window(st, "yaw_unwrapped", t0, t1)
        yaw_t = float(np.median(yw)) if len(yw) else float(interp_state(st, "yaw_unwrapped", t0))
        psi = math.radians(heading + yaw_sign * (yaw_t - yaw_ref))
        pos = pos + int(r[2].split()[1]) / 100.0 * np.array([math.cos(psi), math.sin(psi)])
        track.append((t1, float(pos[0]), float(pos[1])))
    return track


def export_ac1(run_dir, out_dir, yaw_sign=None, camera_offset_m=None, calib=None, positions="imu",
               verbose=True):
    """Кадри в моменти захоплення (мітки "capture") -> журнал обльоту АС Р1.
    positions: "imu" -- положення точок за виконаними кроками з курсом із рискання (точніше:
    до кінця облету 03.11.2026 похибка 0,18-1,08 м проти 0,44-1,92 м за сумою команд);
    "commands" -- як у мітках capture (сума команд)."""
    with open(os.path.join(run_dir, "mission.json"), encoding="utf-8") as f:
        mission = json.load(f)
    with open(os.path.join(run_dir, "marks.json"), encoding="utf-8") as f:
        marks = json.load(f)
    cam = mission.get("camera", {})
    yaw_sign = float(cam.get("yaw_sign", -1.0) if yaw_sign is None else yaw_sign)
    camera_offset_m = float(cam.get("offset_m", 0.0) if camera_offset_m is None else camera_offset_m)
    captures = [m for m in marks if m["kind"] == "capture"]
    ref = next((m for m in marks if m["kind"] == "yaw_ref"), None)
    if not captures or ref is None:
        raise RuntimeError("у записі немає міток capture / yaw_ref (це не місія survey або її перервано до зйомки)")
    if not os.path.exists(os.path.join(run_dir, "frames.csv")):
        decode_run(run_dir, verbose=verbose)
    fr_t, fr_s = [], []
    with open(os.path.join(run_dir, "frames.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fr_t.append(float(row["t"]))
            fr_s.append(float(row["sharpness"]))
    fr_t, fr_s = np.asarray(fr_t), np.asarray(fr_s)
    st = load_state(run_dir)
    yaw_ref = float(interp_state(st, "yaw_unwrapped", ref["t"]))

    picks = []                       # (номер кадру, мітка, висота, рискання Tello)
    for c in captures:
        t0, t1 = c["t"], c["t"] + float(c.get("hold_s", 0.6))
        sel = np.where((fr_t >= t0) & (fr_t <= t1))[0]
        if len(sel) == 0:
            near = int(np.argmin(np.abs(fr_t - 0.5 * (t0 + t1)))) if len(fr_t) else None
            if near is None or abs(fr_t[near] - 0.5 * (t0 + t1)) > 0.5:
                print(f"  точка {c['waypoint']}, поворот {c['k']}: кадрів у вікні немає -- пропущено")
                continue
            sel = np.asarray([near])
        best = int(sel[np.argmax(fr_s[sel])])
        tof = state_window(st, "tof", t0, t1)
        z = (float(np.median(tof)) if len(tof) else float(interp_state(st, "tof", 0.5 * (t0 + t1)))) / 100.0
        # дрон нерухомий у вікні [t0, t1], поворот починається після t1: медіана відліків вікна,
        # а не інтерполяція на час кадру (вона зачепила б наступний відлік, уже під час повороту)
        yaw_win = state_window(st, "yaw_unwrapped", t0, t1)
        yaw_t = float(np.median(yaw_win)) if len(yaw_win) else float(interp_state(st, "yaw_unwrapped", fr_t[best]))
        picks.append((best, c, z + camera_offset_m, yaw_t))

    # знак рискання -- з самого облету: між сусідніми знімками в точці дрон виконав «cw 90»
    steps = [p2[3] - p1[3] for p1, p2 in zip(picks, picks[1:])
             if p1[1]["waypoint"] == p2[1]["waypoint"] and p2[1]["k"] == p1[1]["k"] + 1]
    if steps:
        med = float(np.median(steps))
        detected = -1.0 if med > 45.0 else (1.0 if med < -45.0 else None)
        if detected is not None and detected != yaw_sign:
            print(f"  знак рискання за поворотами облету {detected:+.0f} (у місії {yaw_sign:+.0f}) -- взято з облету")
        if detected is not None:
            yaw_sign = detected
    heading = float(mission.get("heading_deg", 0.0))
    picks = [(b, c, z, (heading + yaw_sign * (y - yaw_ref) + 180.0) % 360.0 - 180.0) for b, c, z, y in picks]

    K = dist = None
    calib = calib or cam.get("calib") or find_calib(run_dir)    # калібрування саме цього дрона (за SN)
    if calib:
        K, dist = _load_calib(calib)
    want = {p[0]: p for p in picks}
    images = {}
    for i, _t, frame in iter_frames(run_dir):
        if i in want:
            if K is not None and cam.get("undistort", True):
                frame = cv2.undistort(frame, K, dist, None, K)
            images[i] = frame
            if len(images) == len(want):
                break

    # формат журналу АС Р1 (ac1_evacmap/flight_log.py) відтворено тут, щоб пакет працював без коду АС
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pred"), exist_ok=True)
    any_img = next(iter(images.values()), None)
    h_px, w_px = (any_img.shape[:2] if any_img is not None else (720, 960))
    hfov = float(cam.get("hfov_deg", HFOV_DEFAULT))
    if K is not None:
        hfov = math.degrees(2.0 * math.atan(w_px / (2.0 * K[0, 0])))
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        drone_info = json.load(f).get("drone")
    meta = {"source": "rmtt", "frame_size": [int(w_px), int(h_px)], "hfov_deg": round(hfov, 2),
            "tilt_deg": float(cam.get("tilt_deg", 10.0)), "height": mission.get("height_m"),
            "room_size": mission.get("room_size", [12.0, 12.0, 3.0]),
            "waypoints": mission.get("waypoints", []), "origin": mission.get("origin"),
            "heading_deg": mission.get("heading_deg", 0.0), "yaw_sign": yaw_sign,
            "camera_offset_m": camera_offset_m, "run": os.path.abspath(run_dir), "drone": drone_info,
            "calib": os.path.abspath(calib) if calib else None, "positions": positions}
    n = 0
    rows = []
    track = forward_track(run_dir, mission, st, yaw_ref, yaw_sign) if positions == "imu" else None
    for best, c, z, yaw in picks:
        if best not in images:
            continue
        x, y = float(c["x"]), float(c["y"])
        if track:
            last = [p for p in track if p[0] <= c["t"]]
            x, y = last[-1][1], last[-1][2]
        cv2.imwrite(os.path.join(out_dir, "frames", f"frame_{n:04d}.png"), images[best])
        rows.append([n, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{yaw:.1f}", int(c["waypoint"])])
        n += 1
    _save_json(os.path.join(out_dir, "meta.json"), meta)
    with open(os.path.join(out_dir, "telemetry.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "z_tof", "yaw_deg", "waypoint"])
        w.writerows(rows)
    if verbose:
        print(f"Журнал АС Р1: {out_dir} -- кадрів {n} з {len(captures)} міток; "
              f"далі: python ../ac1_evacmap/app.py predict --log {out_dir}")
    return {"frames": n, "captures": len(captures), "out": out_dir}


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
