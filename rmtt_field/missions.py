"""Місії натурних дослідів (опис -- JSON у теці missions/).

  bench  -- без зльоту: зв'язок, ідентифікація, телеметрія, передній ToF,
            запис і декодування відео, знак рискання (--yaw-check);
  hover  -- зліт, зависання на заданій висоті, посадка (перший політ);
  line   -- проліт уперед на задану відстань частинами з перевіркою ToF
            (протокол статті про перешкоди: висота 1,4 м, 3,2 м, 20 см/с);
  survey -- протокол АС Р1: точки зависання, у кожній 4 повороти по 90 градусів,
            мітки моментів зйомки; експорт -- postprocess.export_ac1.

Кожен запуск пише теку runs/<дата_час>_<назва>_d<номер>/:
  mission.json, meta.json, marks.json, commands.csv, state.csv, ext_tof.csv,
  video.h264, video_index.csv; після декодування -- video.mp4, frames.csv.
"""
import json
import math
import os
import threading
import time
import traceback

import numpy as np

from drone import DroneError, SafetyStop, _angdiff
from tello_link import VERSION
from safety import MissionAbort

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(HERE, "runs")

DEFAULT_SAFETY = {
    "min_clear_m": 0.5,       # найменший запас до перешкоди попереду після кроку
    "require_tof": True,      # без відповіді переднього ToF рух уперед заборонено
    "tof_max_m": 1.2,         # покази понад цю межу вважаються рівними їй
    "blind_step_m": 0.5,      # крок без ToF, якщо require_tof = false
    "min_battery": 50,        # %, не злітати з меншим зарядом
    "land_battery": 20,       # %, сідати в польоті
    "max_temp_c": 85,         # °C, не злітати з гарячим дроном
    "max_height_m": 2.0,
    "link_timeout_s": 3.0,    # немає телеметрії довше -- переривання (дрон сам сяде через 15 с)
    "on_block": "land",       # перешкода на шляху до точки: land або capture_here
}


def load_mission(path):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    m["safety"] = {**DEFAULT_SAFETY, **m.get("safety", {})}
    cam = m.setdefault("camera", {})
    if cam.get("calib") and not os.path.isabs(cam["calib"]):
        cam["calib"] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), cam["calib"]))
    kind = m.get("type")
    if kind not in ("bench", "hover", "line", "survey"):
        raise ValueError(f"{path}: невідомий тип місії {kind!r}")
    if kind == "survey":
        for key in ("origin", "waypoints", "room_size"):
            if key not in m:
                raise ValueError(f"{path}: для survey потрібне поле {key!r}")
    if m.get("height_m", 1.0) > m["safety"]["max_height_m"]:
        raise ValueError(f"{path}: висота {m['height_m']} м більша за max_height_m")
    return m


class MissionRun:
    def __init__(self, drone, mission, safety, runs_root=RUNS_ROOT):
        self.d = drone
        self.m = mission
        self.S = mission["safety"]
        self.safety = safety
        self.local_abort = threading.Event()
        self.local_reason = ""
        self.marks = []
        self.result = {}
        self.status = "підготовка"
        self.outcome = None
        self.running = False
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.abspath(os.path.join(runs_root, f"{stamp}_{mission['name']}_d{drone.label}"))

    # ------------------------------------------------------------ службове
    def check(self):
        self.safety.check()
        if self.local_abort.is_set():
            raise MissionAbort(self.local_reason)

    def mark(self, kind, **kw):
        self.marks.append({"t": time.time(), "kind": kind, **kw})

    def sleep(self, seconds):
        t_end = time.time() + seconds
        while True:
            self.check()
            left = t_end - time.time()
            if left <= 0:
                return
            time.sleep(min(0.1, left))

    def _monitor(self):
        while self.running:
            if self.d.airborne:
                age, st = self.d.state()
                if age is None or age > self.S["link_timeout_s"]:
                    self._local_abort("втрачено телеметрію дрона")
                elif st.get("bat", 100) < self.S["land_battery"]:
                    self._local_abort(f"низький заряд {st.get('bat'):.0f} %")
            time.sleep(0.5)

    def _local_abort(self, reason):
        if not self.local_abort.is_set():
            self.local_reason = f"дрон {self.d.label}: {reason}"
            self.local_abort.set()
            print(f"\n!!! {self.local_reason}")

    # ------------------------------------------------------------ запуск
    def run(self):
        os.makedirs(self.dir, exist_ok=True)
        _save_json(os.path.join(self.dir, "mission.json"), self.m)
        meta = {"drone": {"label": self.d.label, "ip": self.d.ip, "sn": self.d.sn, "info": self.d.info},
                "mission": self.m["name"], "type": self.m["type"], "software": VERSION, "t_start": time.time(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        st0 = self.d.link.state.stats.get(self.d.ip, [0, 0, 0])[0]
        self.running = True
        self.d.start_recording(self.dir, video=True)
        video = self.m.get("video", {})
        for key, cmd in (("bitrate", "setbitrate"), ("fps", "setfps"), ("resolution", "setresolution")):
            if video.get(key) is not None:
                meta.setdefault("video_settings", {})[key] = self.d.try_send(f"{cmd} {video[key]}")
        mon = threading.Thread(target=self._monitor, daemon=True)
        mon.start()
        try:
            getattr(self, "_" + self.m["type"])()
            self.outcome = "ok"
        except (MissionAbort, SafetyStop, DroneError) as exc:
            self.outcome = f"перервано: {exc}"
        except Exception as exc:  # помилка програми: дрон однаково має сісти
            self.outcome = f"помилка програми: {exc!r}"
            with open(os.path.join(self.dir, "traceback.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        finally:
            if self.d.airborne:
                self._safe_land()
            self.running = False
            self.status = "запис зупиняється"
            self.d.stop_recording()
            self.status = "завершено"
            meta.update({"t_end": time.time(), "outcome": self.outcome, "result": self.result,
                         "state_packets": self.d.link.state.stats.get(self.d.ip, [0, 0, 0])[0] - st0,
                         "abort_reason": self.safety.reason or self.local_reason or None})
            _save_json(os.path.join(self.dir, "marks.json"), self.marks)
            _save_json(os.path.join(self.dir, "meta.json"), meta)
        return self.dir, self.outcome

    # ------------------------------------------------------------ спільні кроки
    def preflight(self, need_tof=True):
        self.status = "передпольотна перевірка"
        st = self.d.wait_state(5.0)
        if st is None:
            raise SafetyStop(f"дрон {self.d.label}: немає телеметрії на порту 8890 "
                             f"(брандмауер Windows блокує Python?)")
        bat, temp = st.get("bat", 0), st.get("temph", 0)
        if bat < self.S["min_battery"]:
            raise SafetyStop(f"дрон {self.d.label}: заряд {bat:.0f} % менший за {self.S['min_battery']} %")
        if temp >= self.S["max_temp_c"]:
            raise SafetyStop(f"дрон {self.d.label}: температура {temp:.0f} °C, дайте охолонути")
        tof = None
        if need_tof:
            tof = self.d.ext_tof_mm()
            if tof is None and self.S["require_tof"]:
                raise SafetyStop(f"дрон {self.d.label}: передній ToF не відповідає (EXT tof?)")
        self.mark("preflight", bat=bat, temph=temp, tof_front_mm=tof)

    def takeoff(self):
        self.status = "зліт"
        self.check()
        self.d.takeoff()
        self.mark("takeoff")
        # справжній RMTT підтверджує зліт через ~9 с, а сідає сам через 15 с без команд (02.11.2026)
        self.d.query("battery?", timeout=2.0, retries=1)
        self.sleep(float(self.m.get("settle_s", 1.5)))

    def climb_to(self, target):
        target = min(float(target), self.S["max_height_m"])
        h = self.d.height_m()
        if h is None:
            raise SafetyStop(f"дрон {self.d.label}: немає показу висоти")
        delta = target - h
        if delta >= 0.2:
            self.d.move("up", min(delta * 100, 500))
        elif delta <= -0.2:
            self.d.move("down", min(-delta * 100, 500))
        self.sleep(float(self.m.get("settle_s", 1.5)))
        self.mark("height", target=target, h=self.d.height_m())

    def land(self):
        self.status = "посадка"
        self.mark("land")
        self.d.land()

    def _landing_now(self):
        """Дрон зараз знижується (vgz <= -2 дм/с протягом останніх 0,6 с)."""
        now = time.time()
        return any(t >= now - 0.6 and st.get("vgz", 0) <= -2
                   for t, st in list(self.d.link.state.history.get(self.d.ip, ())))

    def _on_ground(self):
        """h у стані -- висота, см; на підлозі 0 (tof на RMTT показує ~33 см, тому не за ним)."""
        st = self.d.fresh_state()
        return st is not None and st.get("h", 99) <= 5

    def _safe_land(self):
        self.status = "аварійна посадка"
        self.mark("safe_land")
        # посадку вже запущено ззовні (panic з іншого вікна, автопосадка Tello): «stop» її перериває,
        # а «land» одразу після «stop» RMTT пропускав -- дрон 13 с висів на 30 см (E3, 02.11.2026)
        t_end = time.time() + 8.0
        while time.time() < t_end and self._landing_now() and not self._on_ground():
            time.sleep(0.2)
        if self._on_ground():
            self.mark("landed_external")
            self.d.airborne = False
            return
        for attempt in range(2):
            try:
                self.d.stop()
            except DroneError:
                pass
            time.sleep(0.5)
            try:
                self.d.land()
                return
            except DroneError as exc:
                if self._on_ground():
                    self.d.airborne = False
                    return
                print(f"дрон {self.d.label}: посадка не підтверджена ({exc}), спроба {attempt + 2}")
                time.sleep(1.0)
        print(f"!!! дрон {self.d.label}: ПОСАДКУ НЕ ПІДТВЕРДЖЕНО -- `python app.py panic` або вручну")

    # ------------------------------------------------------------ bench
    def _bench(self):
        secs = float(self.m.get("seconds", 10))
        rep = {"info": self.d.read_info(), "label": self.d.show_label()}
        st = self.d.wait_state(3.0)
        rep["state_alive"] = st is not None
        rep["battery"] = st.get("bat") if st else None
        rep["temph"] = st.get("temph") if st else None
        rep["height_cm_on_ground"] = st.get("tof") if st else None
        self.status = "передній ToF"
        tof = []
        for _ in range(int(self.m.get("tof_samples", 8))):
            tof.append(self.d.ext_tof_mm())
            time.sleep(0.25)
        rep["tof_front_mm"] = tof
        yaw_check = bool(self.m.get("yaw_check"))
        if yaw_check:
            print(f"\nДрон {self.d.label}: ЗАРАЗ поверніть дрон рукою на 90 градусів ЗА годинниковою стрілкою "
                  f"(вигляд згори) і покладіть, у вас {secs:.0f} с")
        self.status = f"запис відео {secs:.0f} с"
        self.mark("bench_record_start", yaw=st.get("yaw") if st else None)
        yaws = []
        t_end = time.time() + secs
        while time.time() < t_end:
            self.check()
            s = self.d.fresh_state()
            if s and "yaw" in s:
                yaws.append(s["yaw"])
            time.sleep(0.1)
        self.mark("bench_record_end", yaw=yaws[-1] if yaws else None)
        if yaw_check and len(yaws) > 1:
            u = np.degrees(np.unwrap(np.radians(yaws)))
            rep["yaw_change_deg"] = round(float(u[-1] - u[0]), 1)
        self.result["bench"] = rep

    # ------------------------------------------------------------ hover
    def _hover(self):
        self.preflight(need_tof=False)
        self.d.set_speed(int(self.m.get("speed_cm_s", 30)))
        self.takeoff()
        self.climb_to(self.m.get("height_m", 1.0))
        self.status = "зависання"
        t_end = time.time() + float(self.m.get("hover_s", 10))
        while time.time() < t_end:
            self.d.query("battery?", timeout=2.0, retries=1)    # дрон сідає сам після 15 с без команд
            self.sleep(min(4.0, max(0.0, t_end - time.time())))
        if self.m.get("spin_check", True):
            self.spin_check()
        self.land()

    def spin_check(self):
        """Знак рискання з польоту: команда «cw 90» і зміна yaw Tello (на стенді оператор
        повертав дрони в різні боки, 29.10.2026). Результат -- result["yaw_sign"] для АС Р1."""
        self.status = "перевірка знака рискання"
        settle = float(self.m.get("settle_s", 1.5))
        y0 = self.d.wait_state(2.0) or {}
        self.d.rotate(90)
        self.sleep(settle)
        y1 = self.d.wait_state(2.0) or {}
        self.d.rotate(-90)
        if "yaw" in y0 and "yaw" in y1:
            delta = (y1["yaw"] - y0["yaw"] + 180.0) % 360.0 - 180.0
            sign = (-1 if delta > 0 else 1) if 60.0 <= abs(delta) <= 120.0 else None
            self.result.update({"yaw_cw90_delta_deg": round(delta, 1), "yaw_sign": sign})
            self.mark("spin_check", delta=delta, yaw_sign=sign)
            print(f"дрон {self.d.label}: «cw 90» змінив рискання Tello на {delta:+.0f}° -> yaw_sign = {sign}")

    # ------------------------------------------------------------ line
    def _line(self):
        self.preflight()
        self.d.set_speed(int(self.m.get("speed_cm_s", 20)))
        self.takeoff()
        self.climb_to(self.m.get("height_m", 1.4))
        self.sleep(float(self.m.get("pre_hover_s", 1.5)))
        self.status = "проліт"
        self.mark("line_start", h=self.d.height_m())
        try:
            done = self.d.forward_safe(float(self.m["distance_m"]), self.S, self.check)
        except SafetyStop as exc:
            self.mark("blocked", reason=str(exc), done=getattr(exc, "done", None))
            raise
        self.mark("line_end", distance_m=done)
        self.result["distance_m"] = done
        self.sleep(float(self.m.get("post_hover_s", 1.0)))
        self.land()

    # ------------------------------------------------------------ survey
    def _heading_now(self):
        """Фактичний курс у кімнаті, град. проти годинникової стрілки від осі x: за рисканням Tello
        відносно yaw_ref (у ту мить дрон дивився за стрілкою зльоту, heading_deg)."""
        self.sleep(0.5)            # телеметрія запізнюється за «ok»: без паузи курс читався ще до повороту
        self.d._wait_still(timeout=2.0)
        yaw = self.d._yaw()
        if yaw is None or self.yaw_ref is None:
            raise SafetyStop(f"дрон {self.d.label}: немає рискання в телеметрії, курс невідомий")
        return _angdiff(self.heading - _angdiff(yaw, self.yaw_ref), 0.0)

    def _face(self, target_deg, tol_deg=3.0):
        """Повернути дрон на курс target_deg за фактичним рисканням, а не за сумою команд.
        E5, 03.11.2026: «cw 90» повертав на 88-89,5°, і до точки 4 курс відходив на 7-24°;
        тричі «ok» на поворот прийшло без повороту, і дрон летів далі під кутом ~80°."""
        for attempt in range(4):                   # до 3 поворотів; err > 0 -- треба за годинниковою
            err = _angdiff(self._heading_now(), target_deg)
            if abs(err) < tol_deg or attempt == 3:
                break
            if attempt and abs(err) > 10.0:
                self.d.note(f"курс відхилено на {err:+.0f}° від потрібного -- доворот")
            self.d.rotate(err)
        self.mark("heading", target=target_deg, err=err)
        if abs(err) > 10.0:
            raise SafetyStop(f"дрон {self.d.label}: не вдалося стати на курс {target_deg:.0f}° "
                             f"(відхилення {err:+.0f}°)")

    def _move_to(self, pos, target, heading_deg):
        delta = np.asarray(target, float) - pos
        dist = float(np.hypot(*delta))
        if dist < 0.2:
            return pos
        self._face(math.degrees(math.atan2(delta[1], delta[0])))   # дрон дивиться в напрямку руху: ToF бачить шлях
        blocked = None
        try:
            done = self.d.forward_safe(dist, self.S, self.check)
        except SafetyStop as exc:
            if not hasattr(exc, "done"):
                raise
            done, blocked = exc.done, exc
        self._face(heading_deg)
        pos = pos + done * delta / dist
        self.mark("moved", target=[float(v) for v in target], done_m=done, x=float(pos[0]), y=float(pos[1]),
                  blocked=str(blocked) if blocked else None)
        if blocked is not None and self.S["on_block"] != "capture_here":
            raise blocked
        return pos

    def _survey(self):
        heading = float(self.m.get("heading_deg", 0.0))
        turns = int(self.m.get("turns", 4))
        settle = float(self.m.get("settle_s", 1.5))
        hold = float(self.m.get("capture_hold_s", 0.6))
        wps = [np.asarray(p, float) for p in self.m["waypoints"]]
        self.preflight()
        self.d.set_speed(int(self.m.get("speed_cm_s", 30)))
        self.takeoff()
        self.climb_to(self.m.get("height_m", 1.2))
        st = self.d.fresh_state() or {}
        self.mark("yaw_ref", yaw_tello=st.get("yaw"))
        self.heading, self.yaw_ref = heading, st.get("yaw")
        pos = np.asarray(self.m["origin"], float)
        n_cap = 0
        for wi, wp in enumerate(wps):
            self.status = f"точка {wi + 1}/{len(wps)}"
            pos = self._move_to(pos, wp, heading)
            for k in range(turns):
                self.sleep(settle)
                s = self.d.fresh_state() or {}
                self.mark("capture", waypoint=wi, k=k, x=float(pos[0]), y=float(pos[1]), hold_s=hold,
                          yaw_tello=s.get("yaw"), tof=s.get("tof"))
                self.sleep(hold)
                n_cap += 1
                self.d.rotate(360.0 / turns)
        self.result["captures"] = n_cap
        self.land()


def run_fleet(assignments, safety, runs_root=RUNS_ROOT, dashboard_s=2.0):
    """assignments: [(drone, mission)]. Місії йдуть паралельно, кожна у своєму потоці."""
    runs = [MissionRun(d, m, safety, runs_root) for d, m in assignments]
    threads = [threading.Thread(target=r.run, daemon=True) for r in runs]
    for th in threads:
        th.start()
    last = 0.0
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(0.2)
            if dashboard_s and time.time() - last >= dashboard_s:
                last = time.time()
                print(" | ".join(_dash(r) for r in runs))
    except KeyboardInterrupt:
        # Ctrl+C (працює на будь-якій ОС): посадка всіх, потім чекаємо, поки місії допишуть записи
        safety.land_all("Ctrl+C")
    for th in threads:
        th.join(timeout=60.0)
    return runs


def _dash(r):
    st = r.d.fresh_state(3.0) or {}
    tof = r.d.last_tof_mm
    tof_txt = "--" if tof is None else ("далеко" if tof >= 8000 else f"{tof / 1000:.2f}")
    return (f"д{r.d.label}: {r.status}, h {st.get('tof', float('nan')) / 100:.2f} м, "
            f"попереду {tof_txt}, бат {st.get('bat', float('nan')):.0f}%, {st.get('temph', float('nan')):.0f}°C")


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
