"""Керування флотом RoboMaster TT для натурних дослідів «Рятівного маршруту».

Команди:
  scan          -- знайти дрони в мережі роутера, оновити fleet.json
  setup-router  -- перевести дрон із прямого підключення на роботу через роутер
  status        -- телеметрія вибраних дронів у консолі (без зльоту)
  bench         -- стендова перевірка без зльоту, підсумок «так / ні» по кожному пункту
  run           -- виконати місію (JSON) одним або кількома дронами
  decode        -- декодувати запис: video.mp4, frames.csv, звіт
  export-ac1    -- запис місії survey -> журнал обльоту АС Р1
  panic         -- посадка (або аварійна зупинка) всіх дронів з іншого вікна
  sim           -- емулятор дронів для репетиції без апаратів

Приклади (один дрон у прямому підключенні, номер 3 на наліпці):
  python app.py bench --ap --label 3 --yaw-check
  python app.py run missions/hover.json --ap --label 3
  python app.py decode runs/<тека>
Флот через роутер:
  python app.py scan
  python app.py bench --drones all
  python app.py run --assign 1=missions/survey_S1.json 2=missions/line_free.json
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tello_link import Link, CMD_PORT, STATE_PORT, VIDEO_PORT, VERSION   # noqa: E402
import fleet as fl                                               # noqa: E402
from missions import load_mission, run_fleet, RUNS_ROOT          # noqa: E402
from safety import Safety                                        # noqa: E402
import postprocess as pp                                         # noqa: E402


def make_link(a):
    try:
        return Link(a.cmd_port, a.state_port, a.video_port, a.bind)
    except OSError as exc:
        raise SystemExit(f"Не вдалося відкрити порти UDP ({exc}). Ймовірно, вже працює інший екземпляр app.py.")


# ---------------------------------------------------------------- scan / setup
def cmd_scan(a):
    link = make_link(a)
    fleet = fl.load_fleet(a.fleet)
    spec = a.hosts or ",".join(fl.local_networks())
    if not spec:
        raise SystemExit("Не знайдено мережевих адрес комп'ютера; вкажіть --hosts 192.168.0.0/24")
    print(f"Пошук дронів: {spec}")
    found = fl.discover(link, fl.parse_hosts(spec), fleet)
    fl.save_fleet(fleet, a.fleet)
    if not found:
        print("Дронів не знайдено. Перевірте: дрони ввімкнено в режимі роутера, комп'ютер у тій самій мережі, "
              "брандмауер дозволяє Python.")
    for entry, info in found:
        print(f"  дрон {entry['label']}: {entry['ip']}, SN {entry['sn']}, {info.get('hardware')}, "
              f"SDK {info.get('sdk')}, заряд {info.get('battery')} %, Wi-Fi {info.get('wifi')}")
    print(f"Збережено: {a.fleet}")
    link.close()


def cmd_setup_router(a):
    link = make_link(a)
    fl.setup_router(link, a.ssid, a.password, a.ip)
    print("Дрон перезавантажиться і підключиться до роутера. Якщо на модулі розширення є перемикач режимів, "
          "переведіть його в положення роботи через роутер. Потім: python app.py scan")
    link.close()


# ---------------------------------------------------------------- status
def cmd_status(a):
    link = make_link(a)
    drones = fl.fleet_drones(link, fl.load_fleet(a.fleet), a.drones, a.ap, a.label)
    for d in drones:
        d.read_info()
        d.show_label()
    print("Ctrl+C -- вихід")
    t_end = time.time() + a.seconds if a.seconds > 0 else float("inf")
    try:
        while time.time() < t_end:
            parts = []
            for d in drones:
                age, st = d.state()
                tof = d.ext_tof_mm()
                rate = link.state.stats.get(d.ip, [0])[0]
                if age is None:
                    parts.append(f"д{d.label}: телеметрії немає")
                    continue
                parts.append(f"д{d.label}: бат {st.get('bat', 0):.0f}%, {st.get('temph', 0):.0f}°C, "
                             f"h {st.get('tof', 0):.0f} см, рискання {st.get('yaw', 0):.0f}°, "
                             f"попереду {'--' if tof is None else ('далеко' if tof >= 8000 else f'{tof} мм')}, "
                             f"пакетів стану {rate}")
            print(" | ".join(parts))
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    link.close()


# ---------------------------------------------------------------- bench
def evaluate_bench(run_dir, rep, dec, yaw_check):
    rows = []

    def add(name, ok, value, hint=""):
        rows.append({"пункт": name, "ok": bool(ok), "значення": value, "що робити": "" if ok else hint})

    info = rep.get("info", {})
    add("зв'язок і SDK", info.get("sdk") is not None, f"SDK {info.get('sdk')}", "оновіть прошивку в RoboMaster")
    add("модель RMTT", info.get("hardware") == "RMTT", info.get("hardware"),
        "не RMTT або старе SDK: команди EXT (ToF, світлодіоди) можуть не працювати")
    rate = dec.get("state_rate_hz")
    add("телеметрія >= 5 Гц", rate is not None and rate >= 5, f"{rate} Гц",
        "дозвольте Python у брандмауері Windows (приватні й загальнодоступні мережі)")
    bat = rep.get("battery")
    add("заряд >= 50 %", bat is not None and bat >= 50, f"{bat} %", "зарядіть батарею")
    temp = rep.get("temph")
    add("температура < 80 °C", temp is not None and temp < 80, f"{temp} °C", "вимкніть дрон на 10 хв")
    tof = [v for v in rep.get("tof_front_mm", []) if v is not None]
    near = [v for v in tof if v < 8000]
    add("передній ToF відповідає", bool(tof), f"{len(tof)} з {len(rep.get('tof_front_mm', []))}",
        "модуль розширення не підключено або не RMTT")
    add("передній ToF бачить долоню", bool(near), f"мін. {min(near)} мм" if near else "лише «далеко»",
        "тримайте долоню 20-40 см перед дроном під час перевірки")
    add("відео декодується", dec.get("frames", 0) > 0, f"{dec.get('frames', 0)} кадрів",
        "немає відеопотоку: брандмауер (порт 11111) або streamon не прийнято")
    fps = dec.get("fps")
    add("частота кадрів >= 20", fps is not None and fps >= 20, f"{fps} к/с", "слабкий Wi-Fi: ближче до роутера, "
        "менше дронів одночасно або setbitrate")
    add("кадри без розривів > 0,5 с", (dec.get("max_gap_s") or 9) <= 0.5, f"{dec.get('max_gap_s')} с",
        "втрати пакетів Wi-Fi")
    add("нумерація кадрів узгоджена", dec.get("count_match", False),
        f"{dec.get('frames')} / {dec.get('frames_in_stream')}", "пошкоджений потік; повторіть запис")
    if yaw_check:
        yc = rep.get("yaw_change_deg")
        ok = yc is not None and 60 <= abs(yc) <= 120
        sign = None if yc is None else (-1 if yc > 0 else 1)
        add("поворот рукою на 90° видно в рисканні", ok, f"{yc}°", "повторіть bench --yaw-check, поверніть на 90°")
        if ok:
            rows[-1]["значення"] += f" -> yaw_sign = {sign} (у місіях survey: camera.yaw_sign)"
    off = 0
    cpath = os.path.join(run_dir, "commands.csv")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            off = sum(1 for row in f if ",# поза чергою," in row)
    rows.append({"пункт": "рядків поза чергою (пропущено)", "ok": True, "значення": off, "що робити": ""})
    with open(os.path.join(run_dir, "bench_report.json"), "w", encoding="utf-8") as f:
        json.dump({"checks": rows, "raw": rep, "decode": dec}, f, ensure_ascii=False, indent=2, default=str)
    return rows


def cmd_bench(a):
    link = make_link(a)
    drones = fl.fleet_drones(link, fl.load_fleet(a.fleet), a.drones, a.ap, a.label)
    safety = Safety()
    for d in drones:
        safety.register(d)
    mission = {"type": "bench", "name": a.name, "seconds": a.seconds, "yaw_check": a.yaw_check,
               "safety": load_mission_defaults()}
    if a.yaw_check:
        print("Під час запису ПОВЕРНІТЬ кожен дрон рукою на 90° за годинниковою стрілкою (вигляд згори).")
    print("Тримайте долоню 20-40 см перед дроном, поки йде перевірка переднього ToF (перші 2 с).")
    runs = run_fleet([(d, mission) for d in drones], safety, a.runs, dashboard_s=0)
    all_ok = True
    for r in runs:
        dec = pp.decode_run(r.dir, verbose=False)
        rows = evaluate_bench(r.dir, r.result.get("bench", {}), dec, a.yaw_check)
        print(f"\n=== Дрон {r.d.label} ({r.d.ip}), {r.outcome}; запис: {r.dir}")
        for row in rows:
            mark = "так" if row["ok"] else "НІ "
            print(f"  [{mark}] {row['пункт']}: {row['значення']}" + (f"  -> {row['що робити']}" if not row["ok"] else ""))
            all_ok &= row["ok"]
    print("\nПІДСУМОК: " + ("усі пункти пройдено" if all_ok else "є пункти «НІ», див. підказки"))
    link.close()


def load_mission_defaults():
    from missions import DEFAULT_SAFETY
    return dict(DEFAULT_SAFETY)


# ---------------------------------------------------------------- run
def cmd_run(a):
    link = make_link(a)
    fleet = fl.load_fleet(a.fleet)
    pairs = []
    if a.assign:
        mapping = dict(item.split("=", 1) for item in a.assign)
        drones = fl.fleet_drones(link, fleet, ",".join(mapping), a.ap, a.label)
        for d in drones:
            pairs.append((d, load_mission(mapping[d.label])))
    else:
        if not a.mission:
            raise SystemExit("Вкажіть файл місії або --assign")
        m = load_mission(a.mission)
        drones = fl.fleet_drones(link, fleet, a.drones, a.ap, a.label)
        if len(drones) > 1 and m["type"] in ("survey", "line"):
            print("УВАГА: одна й та сама траєкторія для кількох дронів -- лише якщо вони стоять у РІЗНИХ приміщеннях.")
        pairs = [(d, m) for d in drones]
    safety = Safety()
    for d, _ in pairs:
        safety.register(d)
        d.read_info()
        d.show_label()
    if not a.no_keyboard:
        safety.start_keyboard()
    if not a.yes:
        print("Місії: " + ", ".join(f"д{d.label} -> {m['name']} ({m['type']})" for d, m in pairs))
        input("Простір вільний, захист гвинтів стоїть, люди за межею? Enter -- старт, Ctrl+C -- скасувати ")
    runs = run_fleet(pairs, safety, a.runs)
    for r in runs:
        print(f"д{r.d.label}: {r.outcome}; запис: {r.dir}")
        if a.decode:
            pp.decode_run(r.dir)
            run_summary(r.dir)
    link.close()


# ---------------------------------------------------------------- обробка
def run_summary(run_dir):
    """Короткий підсумок запису: результат місії, мітки, відновлення після втрат."""
    meta = _read_json(os.path.join(run_dir, "meta.json"), {})
    marks = _read_json(os.path.join(run_dir, "marks.json"), [])
    mission = _read_json(os.path.join(run_dir, "mission.json"), {})
    caps = sum(1 for m in marks if m.get("kind") == "capture")
    line = f"  місія {meta.get('mission')} ({meta.get('type')}), дрон {meta.get('drone', {}).get('label')}: {meta.get('outcome')}"
    if mission.get("type") == "survey":
        want = len(mission.get("waypoints", [])) * int(mission.get("turns", 4))
        line += f"; кадрів зйомки {caps} з {want}"
    blocked = [m for m in marks if m.get("kind") in ("blocked", "moved") and m.get("blocked")]
    if blocked or any(m.get("kind") == "blocked" for m in marks):
        line += "; є зупинка за ToF"
    notes = 0
    cpath = os.path.join(run_dir, "commands.csv")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            notes = sum(1 for row in f if ",# примітка," in row)
    if notes:
        line += f"; відновлень після втрат Wi-Fi {notes}"
    res = meta.get("result") or {}
    if res.get("yaw_cw90_delta_deg") is not None:
        line += f"; «cw 90» -> {res['yaw_cw90_delta_deg']:+.0f}°, yaw_sign = {res.get('yaw_sign')}"
    print(line)


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def cmd_decode(a):
    for pattern in a.runs_dirs:
        for run_dir in sorted(glob.glob(pattern)) or [pattern]:
            pp.decode_run(run_dir)
            run_summary(run_dir)


def cmd_export_ac1(a):
    pp.export_ac1(a.run_dir, a.out, yaw_sign=a.yaw_sign, camera_offset_m=a.camera_offset, calib=a.calib,
                  positions=a.positions)


# ---------------------------------------------------------------- panic
def cmd_panic(a):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)      # окремий сокет: основний процес може зависнути
    fleet = fl.load_fleet(a.fleet)
    ips = [fl.AP_IP] if a.ap else [e["ip"] for e in fleet["drones"]]
    words = ["emergency"] if a.emergency else ["stop", "land"]
    for _ in range(3):
        for ip in ips:
            for w in words:
                sock.sendto(w.encode("ascii"), (ip, 8889))
        time.sleep(0.2)
    print(f"Надіслано «{' + '.join(words)}» на {', '.join(ips)}")
    sock.close()


# ---------------------------------------------------------------- sim
def cmd_sim(a):
    from sim_rmtt import run_emulator
    run_emulator(n=a.n, base=a.base, room=json.loads(a.room) if a.room else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"rmtt_field {VERSION}")
    ap.add_argument("--fleet", default=fl.FLEET_PATH)
    ap.add_argument("--runs", default=RUNS_ROOT, help="тека для записів")
    ap.add_argument("--cmd-port", type=int, default=CMD_PORT)
    ap.add_argument("--state-port", type=int, default=STATE_PORT)
    ap.add_argument("--video-port", type=int, default=VIDEO_PORT)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--ap-ip", default=fl.AP_IP, help="адреса дрона в прямому підключенні (для репетиції з емулятором: 127.0.0.11)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="знайти дрони в мережі")
    s.add_argument("--hosts", default=None, help="192.168.0.0/24 або 127.0.0.11-15; типово -- мережі цього ПК")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("setup-router", help="перевести дрон на роботу через роутер")
    s.add_argument("--ssid", required=True)
    s.add_argument("--password", required=True)
    s.add_argument("--ip", default=fl.AP_IP)
    s.set_defaults(func=cmd_setup_router)

    for name, func, hlp in (("status", cmd_status, "телеметрія в консолі"), ("bench", cmd_bench, "стендова перевірка")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--drones", default="all", help="1,3 або all")
        s.add_argument("--ap", action="store_true", help="один дрон у режимі прямого підключення (192.168.10.1)")
        s.add_argument("--label", default="1", help="номер на наліпці дрона (з --ap)")
        s.add_argument("--seconds", type=float, default=10.0 if name == "bench" else 0.0)
        if name == "bench":
            s.add_argument("--yaw-check", action="store_true", help="перевірити знак рискання поворотом рукою")
            s.add_argument("--name", default="bench", help="назва запису (calib -- для калібрування камери)")
        s.set_defaults(func=func)

    s = sub.add_parser("run", help="виконати місію")
    s.add_argument("mission", nargs="?")
    s.add_argument("--drones", default="1")
    s.add_argument("--assign", nargs="+", help="1=missions/a.json 2=missions/b.json")
    s.add_argument("--ap", action="store_true", help="один дрон у режимі прямого підключення")
    s.add_argument("--label", default="1", help="номер на наліпці дрона (з --ap)")
    s.add_argument("--yes", action="store_true", help="без запитання перед стартом")
    s.add_argument("--no-keyboard", action="store_true")
    s.add_argument("--decode", action="store_true", help="одразу декодувати записи")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("decode", help="декодувати записи")
    s.add_argument("runs_dirs", nargs="+")
    s.set_defaults(func=cmd_decode)

    s = sub.add_parser("export-ac1", help="запис survey -> журнал АС Р1")
    s.add_argument("run_dir")
    s.add_argument("--out", required=True)
    s.add_argument("--yaw-sign", type=float, default=None)
    s.add_argument("--camera-offset", type=float, default=None, help="камера вище ToF-датчика висоти, м")
    s.add_argument("--calib", default=None, help="calib/calib_dN.json цього дрона (типово -- за SN у calib/)")
    s.add_argument("--positions", choices=("imu", "commands"), default="imu",
                   help="положення точок: imu -- кроки з курсом із рискання, commands -- сума команд")
    s.set_defaults(func=cmd_export_ac1)

    s = sub.add_parser("panic", help="посадка всіх з іншого вікна")
    s.add_argument("--emergency", action="store_true", help="зупинити двигуни (дрони падають)")
    s.add_argument("--ap", action="store_true")
    s.set_defaults(func=cmd_panic)

    s = sub.add_parser("sim", help="емулятор дронів")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--base", default="127.0.0.11")
    s.add_argument("--room", default=None, help='JSON: {"size": [8, 6], "boxes": [[x0, y0, x1, y1], ...]}')
    s.set_defaults(func=cmd_sim)

    a = ap.parse_args()
    fl.AP_IP = a.ap_ip
    a.func(a)


if __name__ == "__main__":
    main()
