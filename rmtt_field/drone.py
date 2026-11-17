"""Один дрон RoboMaster TT: команди з підтвердженням, запити, передній
датчик ToF, світлодіоди, безпечні переміщення і запис польоту.

Системи відліку:
  Tello -- рискання за годинниковою стрілкою (вигляд згори) додатне,
           команда "cw N" збільшує кут;
  карта АС Р1 -- x уперед, y ліворуч, z угору, рискання проти годинникової
           стрілки додатне. Перерахунок робить postprocess.export_ac1
           (знак перевіряється на стенді командою bench --yaw-check).
"""
import csv
import math
import queue
import re
import threading
import time

from tello_link import StateLog, VideoRecorder

LABEL_COLORS = {"1": (255, 0, 0), "2": (0, 255, 0), "3": (0, 0, 255),
                "4": (255, 160, 0), "5": (255, 0, 255)}

# цифри 8 x 8 для світлодіодної матриці (1 -- світиться)
GLYPHS = {
    "0": ["00111100", "01100110", "01101110", "01110110", "01100110", "01100110", "00111100", "00000000"],
    "1": ["00011000", "00111000", "00011000", "00011000", "00011000", "00011000", "01111110", "00000000"],
    "2": ["00111100", "01100110", "00000110", "00001100", "00110000", "01100000", "01111110", "00000000"],
    "3": ["00111100", "01100110", "00000110", "00011100", "00000110", "01100110", "00111100", "00000000"],
    "4": ["00001100", "00011100", "00101100", "01001100", "01111110", "00001100", "00001100", "00000000"],
    "5": ["01111110", "01100000", "01111100", "00000110", "00000110", "01100110", "00111100", "00000000"],
    "6": ["00111100", "01100000", "01111100", "01100110", "01100110", "01100110", "00111100", "00000000"],
    "7": ["01111110", "00000110", "00001100", "00011000", "00110000", "00110000", "00110000", "00000000"],
    "8": ["00111100", "01100110", "01100110", "00111100", "01100110", "01100110", "00111100", "00000000"],
    "9": ["00111100", "01100110", "01100110", "00111110", "00000110", "00001100", "00111000", "00000000"],
}

TOF_OUT_OF_RANGE_MM = 8000     # передній ToF повертає 8190/8192, коли перешкоди в межах вимірювання немає


def _angdiff(a, b):
    """a - b, зведене до [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


# Натурні дані 29.10.2026: на команди EXT модуль розширення RMTT часто шле спершу
# «unknown command: keepalive», а за 1-3 мс -- справжню відповідь. Тому відповідь
# приймається лише тоді, коли вона за виглядом підходить до надісланої команди.
_NOISE = re.compile(r"^unknown command:\s*keepalive", re.I)
_TOF_REPLY = re.compile(r"^tof\s+-?\d+", re.I)
_EXT_LINE = re.compile(r"^(tof\s+-?\d+|led ok|matrix ok|mled ok)$", re.I)
# відповідь на керуючу команду; запізніла відповідь на запит (напр. «85» на battery?) нею не буває
_CONTROL_REPLY = re.compile(r"^(ok|error|out of range|forced stop|unknown command)", re.I)
_OK_LIKE = re.compile(r"^(ok|forced stop)$", re.I)


def _accepts(command):
    """Функція: чи може рядок бути відповіддю на command."""
    c = command.strip()
    if c.startswith("EXT tof?"):
        return lambda r: _TOF_REPLY.match(r) is not None
    if c.startswith("EXT"):
        return lambda r: not _NOISE.match(r) and not _TOF_REPLY.match(r)
    if c.endswith("?"):
        return lambda r: not _NOISE.match(r) and not _EXT_LINE.match(r) and not _OK_LIKE.match(r)
    return lambda r: not _NOISE.match(r) and _CONTROL_REPLY.match(r) is not None


class DroneError(RuntimeError):
    """Дрон відповів помилкою."""


class DroneTimeout(DroneError):
    """Дрон не відповів у відведений час."""


class SafetyStop(DroneError):
    """Дію заборонено правилами безпеки (перешкода, батарея, температура)."""


class Drone:
    def __init__(self, link, ip, label="?", sn=""):
        self.link = link
        self.ip = ip
        self.label = str(label)
        self.sn = sn
        self.q = link.cmd.register(ip)
        self.cmd_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.cmd_log = None
        self.tof_log = None
        self.airborne = False
        self.speed_cm_s = 30
        self.info = {}
        self.run_dir = None
        self.last_tof_mm = None
        self.offqueue = 0             # рядків поза чергою (не відповідь на надіслану команду)

    def __repr__(self):
        return f"Drone({self.label}, {self.ip})"

    # ------------------------------------------------------------ журнал
    def _log(self, t0, text, t1, resp):
        with self.log_lock:
            if self.cmd_log is not None:
                self.cmd_log.writerow([f"{t0:.4f}", f"{t1:.4f}" if t1 else "", text,
                                       "TIMEOUT" if resp is None else resp])
                self._cmd_file.flush()

    # ------------------------------------------------------------ команди
    def _flush(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def send(self, text, timeout=7.0, retries=0, expect_ok=True):
        """Команда з очікуванням відповіді. Повтор (retries) -- лише для запитів:
        повтор команди руху після втраченого підтвердження означав би подвійний рух.
        Приймається лише рядок, що може бути відповіддю саме на цю команду (_accepts);
        інші рядки (поза чергою) пишуться в журнал і пропускаються."""
        resp = None
        accept = _accepts(text)
        with self.cmd_lock:
            for _ in range(retries + 1):
                self._flush()
                t0 = time.time()
                self.link.cmd.send(self.ip, text)
                t1, resp = None, None
                deadline = t0 + timeout
                while True:
                    left = deadline - time.time()
                    if left <= 0:
                        break
                    try:
                        t_r, r = self.q.get(timeout=left)
                    except queue.Empty:
                        break
                    if accept(r):
                        t1, resp = t_r, r
                        break
                    self.offqueue += 1
                    self._log(t_r, "# поза чергою", None, f"{r} (очікувалась відповідь на «{text}»)")
                self._log(t0, text, t1, resp)
                if resp is not None:
                    break
        if resp is None:
            raise DroneTimeout(f"дрон {self.label}: немає відповіді на «{text}»")
        if expect_ok and not resp.lower().startswith("ok"):
            raise DroneError(f"дрон {self.label}: «{text}» -> «{resp}»")
        return resp

    def send_nowait(self, text):
        """Без очікування й без блокування (аварійні команди, rc)."""
        t0 = time.time()
        self.link.cmd.send(self.ip, text)
        self._log(t0, text, None, "(без очікування)")

    def query(self, text, timeout=2.0, retries=2):
        return self.send(text, timeout=timeout, retries=retries, expect_ok=False)

    def connect(self):
        self.send("command", timeout=2.0, retries=3)

    def read_info(self):
        for key, cmd in (("sn", "sn?"), ("sdk", "sdk?"), ("hardware", "hardware?"),
                         ("battery", "battery?"), ("wifi", "wifi?")):
            try:
                resp = self.query(cmd)
                self.info[key] = None if resp.lower().startswith("error") else resp
            except DroneError:
                self.info[key] = None
        if self.info.get("sn"):
            self.sn = self.info["sn"]
        return self.info

    def try_send(self, text, timeout=2.0):
        """Необов'язкові налаштування SDK 3.0: помилку лише повертаємо."""
        try:
            return self.send(text, timeout=timeout, retries=1, expect_ok=False)
        except DroneError as exc:
            return f"error: {exc}"

    # ------------------------------------------------------------ модуль розширення
    def ext_tof_mm(self):
        """Передній ToF модуля розширення, мм; None -- немає відповіді."""
        try:
            resp = self.send("EXT tof?", timeout=1.0, retries=2, expect_ok=False)
        except DroneError:
            resp = None
        mm = None
        if resp is not None:
            mm = int(re.findall(r"-?\d+", resp)[-1])      # send прийняв лише «tof N»
        self.last_tof_mm = mm
        with self.log_lock:
            if self.tof_log is not None:
                self.tof_log.write(f"{time.time():.4f},{'' if mm is None else mm}\n")
        return mm

    def front_clearance_m(self, tof_max_m):
        """Вільна відстань попереду, м. Поза межею вимірювання повертається
        tof_max_m (консервативно); None -- датчик не відповів."""
        mm = self.ext_tof_mm()
        if mm is None:
            return None
        if mm >= TOF_OUT_OF_RANGE_MM:
            return tof_max_m
        return min(mm / 1000.0, tof_max_m)

    def led(self, rgb):
        r, g, b = (int(v) for v in rgb)
        return self.try_send(f"EXT led {r} {g} {b}", timeout=1.5)

    def show_label(self):
        """Колір верхнього світлодіода і номер на матриці -- щоб не сплутати дрони."""
        rgb = LABEL_COLORS.get(self.label, (255, 255, 255))
        out = {"led": self.led(rgb)}
        glyph = GLYPHS.get(self.label[-1:])
        if glyph:
            color = "r" if rgb[0] >= max(rgb[1], rgb[2]) else ("b" if rgb[2] > rgb[1] else "p")
            pattern = "".join(color if c == "1" else "0" for row in glyph for c in row)
            out["mled"] = self.try_send(f"EXT mled g {pattern}", timeout=1.5)
        return out

    # ------------------------------------------------------------ стан
    def state(self):
        return self.link.state.get(self.ip)

    def fresh_state(self, max_age=1.5):
        age, st = self.state()
        return st if age is not None and age <= max_age else None

    def wait_state(self, timeout=5.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            st = self.fresh_state()
            if st:
                return st
            time.sleep(0.1)
        return None

    def height_m(self):
        st = self.fresh_state()
        return None if st is None or "tof" not in st else st["tof"] / 100.0

    def alt_cm(self):
        """Поле h стану -- висота відносно зльоту, см; на підлозі 0. Нижній ToF на підлозі RMTT
        показує ~33 см (02.11.2026), тому «сів / злетів» визначається за h, а не за ним."""
        st = self.fresh_state()
        return None if st is None or "h" not in st else float(st["h"])

    # ------------------------------------------------------------ рух
    def set_speed(self, cm_s):
        cm_s = int(max(10, min(100, cm_s)))
        self.send(f"speed {cm_s}", timeout=2.0, retries=2)
        self.speed_cm_s = cm_s

    def takeoff(self):
        self.airborne = True          # з цієї миті посадка обов'язкова навіть після тайм-ауту
        try:
            self.send("takeoff", timeout=12.0)
        except DroneTimeout:
            if self._late_reply("takeoff", 3.0) == "ok":
                return
            h = self.alt_cm()
            if h is not None and h > 40:
                self.note(f"«takeoff»: підтвердження втрачено, висота {h:.0f} см -- зліт виконано")
                return
            if h is not None and h <= 5:
                self.note("«takeoff»: команда не дійшла (дрон на підлозі) -- повтор")
                self.send("takeoff", timeout=12.0)
                return
            raise

    def land(self):
        try:
            resp = self.send("land", timeout=12.0, expect_ok=False)
        except DroneTimeout:
            h = self.alt_cm()
            if h is None or h > 5:
                raise
            resp = "ok (підтвердження втрачено, дрон на підлозі)"
            self.note("«land»: підтвердження втрачено, висота мала -- посадку виконано")
        self.airborne = False
        return resp

    def stop(self):
        return self.send("stop", timeout=3.0, expect_ok=False)

    def emergency(self):
        self.send_nowait("emergency")
        self.airborne = False

    def move(self, direction, cm):
        cm = int(round(cm))
        if direction not in ("forward", "back", "left", "right", "up", "down"):
            raise ValueError(direction)
        if not 20 <= cm <= 500:
            raise ValueError(f"переміщення {cm} см поза межами 20..500")
        cmd = f"{direction} {cm}"
        # тайм-аут + запізніле «ok» + очікування зупинки мають бути < 15 с, інакше Tello сяде сам;
        # справжній крок 62 см при 30 см/с триває 2,1-3,7 с (E5, 03-04.11.2026)
        timeout = 3.0 + 1.6 * cm / max(self.speed_cm_s, 10)
        t0 = time.time()
        try:
            self.send(cmd, timeout=timeout)
            return
        except DroneTimeout:
            pass
        # «ok» на крок у реальних RMTT приходило й після тайм-ауту; недочекане, воно діставалося
        # наступній команді: «cw 82» отримувало «ok» без повороту (E5, 03.11.2026)
        late = self._late_reply(cmd, 2.5)
        if late is not None:
            if late.lower().startswith("ok"):
                return
            raise DroneError(f"дрон {self.label}: «{cmd}» -> «{late}» (із запізненням)")
        self._wait_still(timeout=3.0)
        vertical = direction in ("up", "down")
        moved = self._travelled_cm(t0, time.time(), vertical)
        # інтеграл швидкості в реальних кроках -- 0,56-0,91 від команди (95 % кроків, 415 кроків E5)
        if moved >= 0.5 * cm:
            self.note(f"«{cmd}»: підтвердження втрачено, за швидкістю пройдено ~{moved:.0f} см -- виконано")
            return
        if moved <= 0.15 * cm and self._motion_samples(t0, time.time()) == 0:
            self.note(f"«{cmd}»: підтвердження втрачено, руху не було (~{moved:.0f} см) -- повтор")
            self.send(cmd, timeout=timeout)
            return
        raise DroneError(f"дрон {self.label}: «{cmd}» без підтвердження, за швидкістю ~{moved:.0f} см "
                         f"із {cm}: невідомо, чи виконано -- місію перервано")

    def rotate(self, deg):
        """deg > 0 -- за годинниковою стрілкою (cw), deg < 0 -- проти (ccw)."""
        d = int(round(abs(deg)))
        if d < 1:
            return
        d = min(d, 360)
        cmd = f"{'cw' if deg > 0 else 'ccw'} {d}"
        timeout = 4.0 + d / 40.0
        # телеметрія відстає від «ok» попередньої команди: без паузи y0 -- ще кут попереднього
        # повороту, і відновлення бачило «70° замість 90°» (зміна виконавиці, 02.11.2026)
        self._wait_still(timeout=1.5)
        y0 = self._yaw()
        try:
            self.send(cmd, timeout=timeout)
        except DroneTimeout:
            if self._late_reply(cmd, 2.0) == "ok":
                return
            self._wait_still(timeout=3.0)
            y1 = self._yaw()
            if y0 is None or y1 is None:
                raise
            turned = _angdiff(y1, y0)              # рискання Tello зростає за годинниковою стрілкою
            wanted = d if deg > 0 else -d
            if abs(_angdiff(turned, wanted)) <= 10.0 and d < 350:
                self.note(f"«{cmd}»: підтвердження втрачено, поворот на {turned:.0f}° видно -- виконано")
                return
            if abs(turned) <= 5.0:
                self.note(f"«{cmd}»: підтвердження втрачено, повороту не було -- повтор")
                self.send(cmd, timeout=timeout)
                return
            raise

    def note(self, text):
        print(f"дрон {self.label}: {text}")
        self._log(time.time(), "# примітка", None, text)

    def _yaw(self):
        st = self.fresh_state()
        return None if st is None or "yaw" not in st else float(st["yaw"])

    def _late_reply(self, command, wait_s):
        """Після тайм-ауту ще wait_s секунд чекати відповідь саме на command. Повертає її або None."""
        accept = _accepts(command)
        t_end = time.time() + wait_s
        with self.cmd_lock:
            while True:
                left = t_end - time.time()
                if left <= 0:
                    return None
                try:
                    t_r, r = self.q.get(timeout=left)
                except queue.Empty:
                    return None
                if accept(r):
                    self._log(t_r, "# примітка", None, f"«{command}» -> «{r}» із запізненням "
                                                       f"(після тайм-ауту)")
                    print(f"дрон {self.label}: «{command}» -> «{r}» із запізненням")
                    return r
                self.offqueue += 1
                self._log(t_r, "# поза чергою", None, f"{r} (очікувалась відповідь на «{command}»)")

    def _travelled_cm(self, t0, t1, vertical=False):
        """Інтеграл швидкості з телеметрії, см (vgx, vgy, vgz -- цілі дм/с, у RMTT занижені)."""
        hist = [(t, st) for t, st in list(self.link.state.history.get(self.ip, ())) if t0 <= t <= t1]
        total = 0.0
        for (ta, a), (tb, _b) in zip(hist, hist[1:]):
            v = abs(a.get("vgz", 0)) if vertical else math.hypot(a.get("vgx", 0), a.get("vgy", 0))
            total += v * 10.0 * min(tb - ta, 0.3)
        return total

    def _motion_samples(self, t0, t1, threshold=2.0):
        """Кількість відліків із помітною швидкістю: горизонтальна |v| або |vgz| не менша за threshold.
        У зависанні RMTT показує до 1 дм/с, у кроці 30 см/с -- 1-2 дм/с (E5, 03.11.2026)."""
        hist = list(self.link.state.history.get(self.ip, ()))
        return sum(1 for t, st in hist if t0 <= t <= t1 and
                   max(math.hypot(st.get("vgx", 0), st.get("vgy", 0)), abs(st.get("vgz", 0))) >= threshold)

    def _wait_still(self, timeout=5.0, quiet_s=0.6):
        """Чекати, поки дрон зупиниться (швидкість і рискання сталі протягом quiet_s)."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            now = time.time()
            hist = [st for t, st in list(self.link.state.history.get(self.ip, ())) if t >= now - quiet_s]
            if len(hist) >= 3:
                v = max(max(math.hypot(st.get("vgx", 0), st.get("vgy", 0)), abs(st.get("vgz", 0))) for st in hist)
                yaws = [st.get("yaw", 0) for st in hist]
                if v < 1.5 and max(abs(_angdiff(y, yaws[0])) for y in yaws) < 2.0:
                    return True
            time.sleep(0.1)
        return False

    def forward_safe(self, dist_m, safety, check=lambda: None):
        """Рух уперед частинами з перевіркою переднього ToF перед кожною.
        Повертає пройдену (за командами) відстань; залишок < 0,2 м не виконується."""
        done = 0.0
        while dist_m - done >= 0.2 - 1e-9:
            check()
            clear = self.front_clearance_m(safety["tof_max_m"])
            if clear is None:
                if safety["require_tof"]:
                    exc = SafetyStop(f"дрон {self.label}: передній ToF не відповідає")
                    exc.done = done
                    raise exc
                step_max = safety["blind_step_m"]
            else:
                step_max = clear - safety["min_clear_m"]
            step_max = min(step_max, 5.0)
            if step_max < 0.2 - 1e-9:
                exc = SafetyStop(f"дрон {self.label}: перешкода попереду на {clear:.2f} м, "
                                 f"пройдено {done:.2f} з {dist_m:.2f} м")
                exc.done = done
                raise exc
            rest = dist_m - done
            step = rest
            if rest > step_max:
                # рівні частини: інакше залишок < 0,2 м (найменша команда Tello) не виконався б
                step = rest / math.ceil(rest / step_max - 1e-9)
                if step < 0.2:
                    step = step_max
            cm = min(int(round(step * 100)), int(math.floor(step_max * 100 + 1e-6)))
            self.move("forward", cm)
            done += cm / 100.0
        return done

    # ------------------------------------------------------------ запис
    def start_recording(self, run_dir, video=True):
        import os
        self.run_dir = run_dir
        self._cmd_file = open(os.path.join(run_dir, "commands.csv"), "w", encoding="utf-8", newline="")
        self.cmd_log = csv.writer(self._cmd_file)
        self.cmd_log.writerow(["t_send", "t_resp", "command", "response"])
        self.tof_log = open(os.path.join(run_dir, "ext_tof.csv"), "w", encoding="utf-8", newline="")
        self.tof_log.write("t,tof_mm\n")
        self.link.state.attach(self.ip, StateLog(os.path.join(run_dir, "state.csv")))
        if video:
            self.link.video.attach(self.ip, VideoRecorder(os.path.join(run_dir, "video.h264"),
                                                          os.path.join(run_dir, "video_index.csv")))
            self.send("streamon", timeout=5.0, retries=1)

    def stop_recording(self):
        try:
            self.send("streamoff", timeout=3.0, retries=1, expect_ok=False)
        except DroneError:
            pass
        time.sleep(0.3)
        self.link.video.detach(self.ip)
        self.link.state.detach(self.ip)
        with self.log_lock:
            if self.cmd_log is not None:
                self._cmd_file.close()
                self.cmd_log = None
            if self.tof_log is not None:
                self.tof_log.close()
                self.tof_log = None
