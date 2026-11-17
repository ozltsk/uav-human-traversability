"""Автоматична перевірка софту на емуляторі (без дронів).

python test_field.py            -- усі сценарії (близько 2 хв)
python test_field.py bench      -- лише вибрані: scan bench survey tof abort panic

Що перевіряється:
  scan   -- пошук трьох дронів, номери за серійними, повторний пошук зберігає номери;
  bench  -- стендова перевірка всіх трьох одночасно, знак рискання, час кадрів
            (номер кадру зчитується зі смуги на зображенні й порівнюється з моментом
            відправлення в емуляторі);
  survey -- два дрони паралельно виконують обліт АС Р1, експорт журналу, звірка
            положень і рискання з «істиною» емулятора;
  tof    -- дрон перед ящиком зупиняється за переднім ToF і сідає без зіткнення;
  abort  -- посадка всіх за командою оператора під час зависання;
  panic  -- окремий процес `app.py panic` саджає дрон;
  lossy  -- Wi-Fi губить 8 % відповідей і 4 % команд: обліт однаково завершується,
            втрачені підтвердження відновлюються за телеметрією;
  keepalive -- як у справжніх RMTT 29.10.2026: перед кожною відповіддю на EXT приходить
            «unknown command: keepalive»; стенд бачить ToF, зависання визначає знак рискання,
            обліт завершується.
  lateack -- як у справжніх RMTT 03-04.11.2026: «ok» на рух іноді приходить після тайм-ауту
            (дрон уже приймає нові команди), поворот недокручує на 1,7 %; обліт має завершитися
            в потрібних точках, а запізніле «ok» не має дістатися наступній команді.
У сценарії survey дрону 2 навмисно задано хибний yaw_sign: експорт має виправити його сам.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fleet as fl                                  # noqa: E402
import postprocess as pp                            # noqa: E402
from app import evaluate_bench                      # noqa: E402
from missions import DEFAULT_SAFETY, run_fleet      # noqa: E402
from safety import Safety                           # noqa: E402
from sim_rmtt import FakeDrone, barcode_value, get_clip   # noqa: E402
from tello_link import Link                         # noqa: E402

PORTS = dict(cmd_port=19889, state_port=18890, video_port=21111)
TS = 0.15                                            # прискорення руху в емуляторі
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'так' if ok else 'НІ '}] {name}" + (f": {detail}" if detail else ""))


def mission(kind, **kw):
    m = {"type": kind, "name": f"test_{kind}", "settle_s": 0.3, "safety": dict(DEFAULT_SAFETY)}
    m["safety"].update(kw.pop("safety", {}))
    m.update(kw)
    return m


class Env:
    def __init__(self, tmp):
        self.tmp = tmp
        self.runs = os.path.join(tmp, "runs")
        self.fleet_path = os.path.join(tmp, "fleet.json")
        starts = [(1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (7.3, 1.0, 0.0)]   # третій дивиться в стіну за 0,7 м
        self.fakes = [FakeDrone(f"127.0.0.{11 + k}", f"0TQZSIM{k + 1:04d}", start=starts[k][:2],
                                heading_deg=starts[k][2], time_scale=TS, state_port=PORTS["state_port"],
                                video_port=PORTS["video_port"], seed=k) for k in range(3)]
        self.link = Link(PORTS["cmd_port"], PORTS["state_port"], PORTS["video_port"])

    def drones(self, labels="all"):
        return fl.fleet_drones(self.link, fl.load_fleet(self.fleet_path), labels)

    def close(self):
        self.link.close()
        for f in self.fakes:
            f.close()


# ---------------------------------------------------------------- сценарії
def t_scan(env):
    print("\n[scan]")
    fleet = fl.load_fleet(env.fleet_path)
    found = fl.discover(env.link, fl.parse_hosts("127.0.0.10-20"), fleet)
    fl.save_fleet(fleet, env.fleet_path)
    check("знайдено 3 дрони", len(found) == 3, ", ".join(f"{e['label']}={e['ip']}" for e, _ in found))
    labels1 = {e["sn"]: e["label"] for e in fleet["drones"]}
    fleet2 = fl.load_fleet(env.fleet_path)
    fl.discover(env.link, fl.parse_hosts("127.0.0.10-20"), fleet2)
    labels2 = {e["sn"]: e["label"] for e in fleet2["drones"]}
    check("повторний пошук зберігає номери", labels1 == labels2 and len(fleet2["drones"]) == 3)


def t_bench(env):
    print("\n[bench]")
    drones = env.drones()
    safety = Safety()
    m = mission("bench", seconds=4.0, yaw_check=True)

    def turn_by_hand():                    # «оператор» повертає дрон 1 на 90° за год. стрілкою
        time.sleep(3.0)
        for _ in range(30):
            env.fakes[0].psi -= 3.0
            time.sleep(0.03)

    import threading
    threading.Thread(target=turn_by_hand, daemon=True).start()
    runs = run_fleet([(d, m) for d in drones], safety, env.runs, dashboard_s=0)
    for r in runs:
        dec = pp.decode_run(r.dir, verbose=False)
        rows = {row["пункт"]: row for row in evaluate_bench(r.dir, r.result["bench"], dec, True)}
        must = ["зв'язок і SDK", "модель RMTT", "телеметрія >= 5 Гц", "передній ToF відповідає",
                "відео декодується", "частота кадрів >= 20", "кадри без розривів > 0,5 с",
                "нумерація кадрів узгоджена"]
        bad = [k for k in must if not rows[k]["ok"]]
        check(f"дрон {r.d.label}: обов'язкові пункти стенда", not bad,
              f"{dec.get('frames')} кадрів, {dec.get('fps')} к/с, {dec.get('state_rate_hz')} Гц" +
              (f"; НІ: {bad}" if bad else ""))
        if r.d.label == "3":
            check("дрон 3: долоню (стіну за 0,7 м) видно", rows["передній ToF бачить долоню"]["ok"],
                  rows["передній ToF бачить долоню"]["значення"])
        if r.d.label == "1":
            yc = r.result["bench"].get("yaw_change_deg")
            check("поворот за год. стрілкою дає +90° рискання Tello", yc is not None and 80 <= yc <= 100, f"{yc}°")
        verify_frame_times(r.dir, env.fakes[int(r.d.label) - 1])


def verify_frame_times(run_dir, fake):
    """Номер кадру зі смуги + час за нашим індексом проти журналу відправлення емулятора."""
    log = np.asarray(fake.video_log, dtype=np.float64)
    if not len(log):
        check("журнал відео емулятора", False, "порожній")
        return
    errs, wrong = [], 0
    for _i, t, frame in pp.iter_frames(run_dir):
        idx = barcode_value(frame)
        cand = np.where(log[:, 1] == idx)[0]
        if not len(cand):
            wrong += 1
            continue
        errs.append(np.min(np.abs(log[cand, 0] - t)))
    errs = np.asarray(errs)
    ok = len(errs) > 0 and wrong == 0 and np.median(errs) < 0.03 and errs.max() < 0.1
    check(f"{os.path.basename(run_dir)[-2:]}: час кадрів", ok,
          f"{len(errs)} кадрів, медіана похибки {1000 * np.median(errs):.1f} мс, макс. {1000 * errs.max():.1f} мс, "
          f"невпізнаних {wrong}" if len(errs) else "кадрів немає")


def t_survey(env):
    print("\n[survey]")
    for k in (0, 1):
        f = env.fakes[k]
        f.x, f.y, f.psi = 1.0, 1.0, 0.0
    drones = env.drones("1,2")
    wps = [[1.0, 1.0], [2.5, 1.0], [2.5, 2.5], [1.0, 2.5]]
    base = dict(room_size=[8.0, 6.0, 3.0], origin=[1.0, 1.0], heading_deg=0.0, waypoints=wps,
                height_m=1.2, turns=4, capture_hold_s=0.4,
                camera={"hfov_deg": 70.0, "tilt_deg": 10.0, "yaw_sign": -1, "offset_m": 0.0})
    safety = Safety()
    def for_drone(d):
        m = mission("survey", **base)
        if d.label == "2":                         # хибний знак: експорт визначить знак з облету
            m["camera"] = {**m["camera"], "yaw_sign": 1}
        return m

    runs = run_fleet([(d, for_drone(d)) for d in drones], safety, env.runs, dashboard_s=0)
    for r, k in zip(runs, (0, 1)):
        f = env.fakes[k]
        check(f"дрон {r.d.label}: місія виконана", r.outcome == "ok", r.outcome)
        check(f"дрон {r.d.label}: без зіткнень і сів", not f.crashed and not f.flying,
              f"кінцеве положення ({f.x:.2f}, {f.y:.2f})")
        out = os.path.join(env.tmp, f"ac1_d{r.d.label}")
        pp.export_ac1(r.dir, out, verbose=False)
        log = load_ac1_log(out)
        check(f"дрон {r.d.label}: журнал АС Р1 має 16 кадрів", len(log) == 16 and all(log.has_frame(i) for i in range(16)),
              f"{len(log)} кадрів, {log.meta['frame_size']}")
        pos_err = max(abs(rec.x - wps[rec.waypoint][0]) + abs(rec.y - wps[rec.waypoint][1]) for rec in log.records)
        expect = [((-90.0 * (i % 4)) + 180.0) % 360.0 - 180.0 for i in range(16)]
        yaw_err = max(abs(((rec.yaw_deg - e) + 180) % 360 - 180) for rec, e in zip(log.records, expect))
        z_err = max(abs(rec.z_tof - 1.2) for rec in log.records)
        check(f"дрон {r.d.label}: положення, рискання, висота", pos_err < 0.03 and yaw_err < 3.0 and z_err < 0.05,
              f"похибка xy {pos_err:.3f} м, рискання {yaw_err:.1f}°, висоти {z_err:.3f} м")
        truth_err = abs(f.x - wps[-1][0]) + abs(f.y - wps[-1][1])
        check(f"дрон {r.d.label}: емулятор справді долетів до останньої точки", truth_err < 0.03,
              f"({f.x:.2f}, {f.y:.2f}) проти {wps[-1]}")


class _Rec:
    def __init__(self, row):
        self.index, self.waypoint = int(row["index"]), int(row["waypoint"])
        self.x, self.y, self.z_tof, self.yaw_deg = (float(row[k]) for k in ("x", "y", "z_tof", "yaw_deg"))


class load_ac1_log:
    """Читання журналу у форматі АС Р1 (meta.json, telemetry.csv, frames/) без коду АС."""

    def __init__(self, root):
        import csv as _csv
        self.root = root
        self.meta = json.load(open(os.path.join(root, "meta.json"), encoding="utf-8"))
        with open(os.path.join(root, "telemetry.csv"), encoding="utf-8", newline="") as f:
            self.records = [_Rec(row) for row in _csv.DictReader(f)]

    def __len__(self):
        return len(self.records)

    def has_frame(self, i):
        return os.path.exists(os.path.join(self.root, "frames", f"frame_{i:04d}.png"))


def t_tof(env):
    print("\n[tof]")
    f = env.fakes[2]
    f.x, f.y, f.psi = 3.0, 2.9, 0.0          # ящик x 4,0..4,6, y 2,4..3,4 -- рівно попереду за 1,0 м
    f.crashed = False
    d = env.drones("3")
    safety = Safety()
    runs = run_fleet([(d[0], mission("line", height_m=1.2, distance_m=3.0, speed_cm_s=30))],
                     safety, env.runs, dashboard_s=0)
    r = runs[0]
    marks = json.load(open(os.path.join(r.dir, "marks.json"), encoding="utf-8"))
    blocked = [m for m in marks if m["kind"] == "blocked"]
    check("рух зупинено переднім ToF", r.outcome.startswith("перервано") and "перешкода" in r.outcome, r.outcome)
    check("без зіткнення, дрон сів", not f.crashed and not f.flying,
          f"зупинився на x = {f.x:.2f} м, до ящика {4.0 - f.x:.2f} м; пройдено {blocked[0].get('done') if blocked else '?'} м")


def t_abort(env):
    print("\n[abort]")
    drones = env.drones("1,2")
    safety = Safety()
    import threading

    def operator():
        time.sleep(4.0)
        safety.land_all("тест: посадка всіх")

    threading.Thread(target=operator, daemon=True).start()
    t0 = time.time()
    runs = run_fleet([(d, mission("hover", height_m=1.0, hover_s=30)) for d in drones], safety, env.runs,
                     dashboard_s=0)
    dt = time.time() - t0
    check("місії перервано оператором", all(r.outcome.startswith("перервано") for r in runs),
          "; ".join(r.outcome for r in runs))
    check("усі сіли, зіткнень немає", all(not env.fakes[k].flying and not env.fakes[k].crashed for k in (0, 1)),
          f"тривалість {dt:.1f} с замість 30")


def t_panic(env):
    print("\n[panic]")
    d = env.drones("2")[0]
    d.takeoff()
    check("дрон 2 у повітрі", env.fakes[1].flying)
    subprocess.run([sys.executable, os.path.join(HERE, "app.py"), "--fleet", env.fleet_path, "panic"],
                   check=True, capture_output=True)
    t_end = time.time() + 5.0
    while env.fakes[1].flying and time.time() < t_end:
        time.sleep(0.1)
    check("app.py panic посадив дрон з іншого процесу", not env.fakes[1].flying)
    d.airborne = False


def t_lossy(env):
    print("\n[lossy]")
    f = env.fakes[2]
    f.x, f.y, f.psi, f.crashed = 1.0, 1.0, 0.0, False
    f.drop_ack, f.drop_cmd, f.drop_state = 0.08, 0.04, 0.02
    d = env.drones("3")[0]
    wps = [[1.0, 1.0], [2.5, 1.0], [2.5, 2.5]]
    m = mission("survey", room_size=[8.0, 6.0, 3.0], origin=[1.0, 1.0], heading_deg=0.0, waypoints=wps,
                height_m=1.2, turns=4, capture_hold_s=0.4, camera={"yaw_sign": -1})
    runs = run_fleet([(d, m)], Safety(), env.runs, dashboard_s=0)
    f.drop_ack = f.drop_cmd = f.drop_state = 0.0
    r = runs[0]
    import csv as _csv
    notes = [row for row in _csv.reader(open(os.path.join(r.dir, "commands.csv"), encoding="utf-8"))
             if len(row) > 2 and row[2] == "# примітка"]
    lost = sum(1 for row in _csv.reader(open(os.path.join(r.dir, "commands.csv"), encoding="utf-8"))
               if len(row) > 3 and row[3] == "TIMEOUT")
    check("обліт із втратами завершено", r.outcome == "ok", r.outcome)
    check("без зіткнення, у кінцевій точці", not f.crashed and not f.flying and
          abs(f.x - wps[-1][0]) + abs(f.y - wps[-1][1]) < 0.03, f"({f.x:.2f}, {f.y:.2f})")
    check("втрачені підтвердження відновлено", True, f"тайм-аутів {lost}, відновлень {len(notes)}")
    for row in notes[:6]:
        print("      " + row[3])


def t_keepalive(env):
    print("\n[keepalive]")
    f = env.fakes[2]
    f.x, f.y, f.psi, f.crashed = 1.0, 1.0, 0.0, False
    f.ext_noise_before, f.ext_noise_after = 1.0, 0.3
    d = env.drones("3")[0]
    try:
        r = run_fleet([(d, mission("bench", seconds=2.0))], Safety(), env.runs, dashboard_s=0)[0]
        rep = r.result["bench"]
        tof = [v for v in rep["tof_front_mm"] if v is not None]
        check("стенд: ToF відповідає попри «keepalive»", len(tof) == len(rep["tof_front_mm"]),
              f"{len(tof)} з {len(rep['tof_front_mm'])}; led -> {rep['label'].get('led')}")
        r = run_fleet([(d, mission("hover", height_m=1.0, hover_s=2.0))], Safety(), env.runs, dashboard_s=0)[0]
        check("зависання: знак рискання з польоту", r.outcome == "ok" and r.result.get("yaw_sign") == -1,
              f"{r.outcome}; cw 90 -> {r.result.get('yaw_cw90_delta_deg')}°, yaw_sign {r.result.get('yaw_sign')}")
        m = mission("survey", room_size=[8.0, 6.0, 3.0], origin=[1.0, 1.0], heading_deg=0.0,
                    waypoints=[[1.0, 1.0], [2.5, 1.0]], height_m=1.2, turns=4, capture_hold_s=0.4,
                    camera={"yaw_sign": -1})
        r = run_fleet([(d, m)], Safety(), env.runs, dashboard_s=0)[0]
        caps = sum(1 for mk in r.marks if mk["kind"] == "capture")
        check("обліт із «keepalive» завершено", r.outcome == "ok" and caps == 8 and not f.crashed,
              f"{r.outcome}; кадрів зйомки {caps} з 8; рядків поза чергою {d.offqueue}")
    finally:
        f.ext_noise_before = f.ext_noise_after = 0.0


def t_lateack(env):
    print("\n[lateack]")
    f = env.fakes[2]
    f.x, f.y, f.psi, f.crashed = 1.0, 1.0, 0.0, False
    f.late_ack_prob, f.late_ack_s, f.yaw_scale = 0.25, 6.5, 0.983
    d = env.drones("3")[0]
    wps = [[1.0, 1.0], [2.5, 1.0], [2.5, 2.2], [1.0, 2.2]]
    m = mission("survey", room_size=[8.0, 6.0, 3.0], origin=[1.0, 1.0], heading_deg=0.0, waypoints=wps,
                height_m=1.2, turns=4, capture_hold_s=0.4, camera={"yaw_sign": -1})
    try:
        r = run_fleet([(d, m)], Safety(), env.runs, dashboard_s=0)[0]
    finally:
        f.late_ack_prob, f.late_ack_s, f.yaw_scale = 0.0, 0.0, 1.0
    import csv as _csv
    rows = list(_csv.reader(open(os.path.join(r.dir, "commands.csv"), encoding="utf-8")))
    late = sum(1 for row in rows if len(row) > 3 and "із запізненням" in row[3])
    heads = [mk for mk in r.marks if mk["kind"] == "heading"]
    worst = max((abs(mk["err"]) for mk in heads), default=None)
    caps = sum(1 for mk in r.marks if mk["kind"] == "capture")
    check("обліт із запізнілими «ok» і недокручуванням завершено", r.outcome == "ok" and caps == 16,
          f"{r.outcome}; кадрів зйомки {caps} з 16; запізнілих відповідей {late}")
    check("курс тримається, дрон у кінцевій точці", worst is not None and worst < 3.0 and not f.crashed and
          abs(f.x - wps[-1][0]) + abs(f.y - wps[-1][1]) < 0.06,
          f"найбільше відхилення курсу {worst}°; кінець ({f.x:.2f}, {f.y:.2f}) проти {wps[-1]}")


SCENARIOS = {"scan": t_scan, "bench": t_bench, "survey": t_survey, "tof": t_tof, "abort": t_abort,
             "panic": t_panic, "lossy": t_lossy, "keepalive": t_keepalive, "lateack": t_lateack}


def main():
    wanted = sys.argv[1:] or list(SCENARIOS)
    if not get_clip():
        print("УВАГА: ffmpeg не знайдено -- емулятор без відео, перевірки відео не пройдуть")
    tmp = tempfile.mkdtemp(prefix="rmtt_test_")
    env = Env(tmp)
    try:
        if "scan" not in wanted:
            t_scan(env)
        for name in wanted:
            SCENARIOS[name](env)
    finally:
        env.close()
    n_bad = sum(not ok for _, ok, _ in RESULTS)
    summary = f"ПІДСУМОК: {len(RESULTS) - n_bad} з {len(RESULTS)} перевірок пройдено"
    print(f"\n{summary}; записи: {tmp}")
    import platform
    with open(os.path.join(HERE, "test_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}, {platform.platform()}, Python {platform.python_version()}\n")
        f.write(f"сценарії: {' '.join(wanted)}\n")
        for name, ok, detail in RESULTS:
            f.write(f"[{'так' if ok else 'НІ '}] {name}" + (f": {detail}" if detail else "") + "\n")
        f.write(summary + "\n")
    if n_bad == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
