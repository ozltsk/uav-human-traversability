"""Емулятор RoboMaster TT для репетиції без апаратів і для автоматичних тестів.

Кожен «дрон» має окрему адресу loopback (127.0.0.11, 127.0.0.12, ...) і порт 8889.
Він відповідає на команди SDK 3.0, шле стан 10 разів на секунду і відео H.264
30 к/с на адресу того, хто надіслав "command". Рух триває відстань/швидкість
у прямокутному приміщенні з ящиками; передній ToF -- промінь до найближчої
стіни чи ящика. Зіткнення -- «аварія»: дрон падає, прапорець crashed.
Дрони одне одного не бачать (кожен у своєму приміщенні).

Відео -- синтетичний ролик, у верхній смузі якого номер кадру закодовано
16 квадратами (білий = 1). За ним тест перевіряє час кожного кадру.
Ролик кодує ffmpeg (libx264); без ffmpeg емулятор працює без відео.
"""
import math
import os
import random
import shutil
import socket
import subprocess
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_sim_cache")
CLIP_FRAMES = 90
CLIP_SIZE = (960, 720)
BITS = 16
PKT = 1460

DEFAULT_ROOM = {"size": [8.0, 6.0], "boxes": [[4.0, 2.4, 4.6, 3.4]]}


# ---------------------------------------------------------------- відеоролик
def barcode_value(frame_bgr):
    """Номер кадру зі смуги квадратів (для тестів)."""
    h, w = frame_bgr.shape[:2]
    cell = w / BITS
    y = int(0.04 * h)
    v = 0
    for b in range(BITS):
        x = int((b + 0.5) * cell)
        if frame_bgr[y, x].mean() > 128:
            v |= 1 << (BITS - 1 - b)
    return v


def make_clip(path, n=CLIP_FRAMES, size=CLIP_SIZE):
    import cv2
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = size
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
           "-profile:v", "baseline", "-bf", "0", "-g", "30", "-b:v", "2500k",
           "-x264opts", "repeat-headers=1", "-pix_fmt", "yuv420p", "-f", "h264", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    yy, xx = np.mgrid[0:h, 0:w]
    cell = w // BITS
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = ((xx + 4 * i) % 256).astype(np.uint8)
        img[..., 1] = ((yy * 0.35 + 60) % 256).astype(np.uint8)
        img[..., 2] = 90
        cv2.rectangle(img, (0, 0), (w, int(0.08 * h)), (0, 0, 0), -1)
        for b in range(BITS):
            if (i >> (BITS - 1 - b)) & 1:
                cv2.rectangle(img, (b * cell + 4, 4), ((b + 1) * cell - 4, int(0.08 * h) - 4), (255, 255, 255), -1)
        cv2.putText(img, f"RMTT SIM frame {i}", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return proc.returncode == 0 and os.path.getsize(path) > 0


def clip_access_units(path):
    """Ролик -> список (номер кадру, байти кадру разом із SPS/PPS перед ним)."""
    from postprocess import nal_scan
    with open(path, "rb") as f:
        data = f.read()
    sps, pics = nal_scan(data)
    bounds = []
    for i, p in enumerate(pics):
        prev = pics[i - 1] if i else -1
        cands = [s for s in sps if prev < s < p]
        bounds.append(min(cands) if cands else p)
    bounds[0] = 0
    bounds.append(len(data))
    return [(i, data[bounds[i]:bounds[i + 1]]) for i in range(len(pics))]


_clip_lock = threading.Lock()
_clip_cache = None


def get_clip():
    global _clip_cache
    with _clip_lock:
        if _clip_cache is None:
            path = os.path.join(CACHE, "clip.h264")
            if not os.path.exists(path) and not make_clip(path):
                _clip_cache = []
            else:
                _clip_cache = clip_access_units(path)
        return _clip_cache


# ---------------------------------------------------------------- дрон
class FakeDrone:
    def __init__(self, ip, sn, room=None, start=(1.0, 1.0), heading_deg=0.0, time_scale=1.0,
                 state_port=8890, video_port=11111, drop=0.0, seed=0, battery=90, tof_range_m=1.5,
                 drop_ack=None, drop_cmd=None, drop_state=None, drop_video=None,
                 ext_noise_before=0.0, ext_noise_after=0.0, late_ack_prob=0.0, late_ack_s=0.0, yaw_scale=1.0):
        self.ip, self.sn = ip, sn
        self.room = room or DEFAULT_ROOM
        self.x, self.y = float(start[0]), float(start[1])
        self.psi0 = self.psi = float(heading_deg)       # курс у системі приміщення, проти год. стрілки
        self.z = 0.0
        self.flying = False
        self.crashed = False
        self.battery = float(battery)
        self.temp = 55.0
        self.speed = 30.0
        self.time_scale = time_scale
        self.state_port, self.video_port = state_port, video_port
        self.tof_range_m = tof_range_m
        self.rng = random.Random(seed)
        # імовірності втрат: відповіді, команди до дрона, рядки стану, пакети відео
        self.drop_ack = drop if drop_ack is None else drop_ack
        self.drop_cmd = drop if drop_cmd is None else drop_cmd
        self.drop_state = drop if drop_state is None else drop_state
        self.drop_video = drop if drop_video is None else drop_video
        self.vel = (0.0, 0.0, 0.0)     # швидкість, дм/с (для vgx, vgy, vgz у стані)
        # як у справжніх RMTT (29.10.2026): «unknown command: keepalive» до або після відповіді на EXT
        self.ext_noise_before = ext_noise_before
        self.ext_noise_after = ext_noise_after
        # як у справжніх RMTT (03-04.11.2026): «ok» на рух іноді приходить через late_ack_s після кінця руху
        # (дрон уже приймає нові команди), поворот недокручує: «cw 90» -> 88-89,5° (yaw_scale ~0,983)
        self.late_ack_prob, self.late_ack_s, self.yaw_scale = late_ack_prob, late_ack_s, yaw_scale
        self.client_ip = None
        self.streaming = False
        self.busy = False
        self.interrupt = threading.Event()
        self.lock = threading.Lock()
        self.t_takeoff = None
        self.events = []               # (t, команда, відповідь) -- для тестів
        self.video_log = []            # (t відправлення останнього пакета, номер кадру ролика)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, 8889))
        self.sock.settimeout(0.2)
        if os.name == "nt" and hasattr(socket, "SIO_UDP_CONNRESET"):
            self.sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.out.bind((ip, 0))         # стан і відео -- з адреси дрона, як у справжнього апарата
        self._run = True
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._cmd_loop, self._state_loop)]
        for th in self.threads:
            th.start()

    # ------------------------------------------------------------ геометрія
    @property
    def yaw_tello(self):
        y = -(self.psi - self.psi0)
        return (y + 180.0) % 360.0 - 180.0

    def _inside_obstacle(self, x, y, r=0.12):
        sx, sy = self.room["size"]
        if x < r or y < r or x > sx - r or y > sy - r:
            return True
        for x0, y0, x1, y1 in self.room["boxes"]:
            if x0 - r < x < x1 + r and y0 - r < y < y1 + r:
                return True
        return False

    def front_distance(self):
        """Промінь у напрямку курсу до першої перешкоди, м."""
        c, s = math.cos(math.radians(self.psi)), math.sin(math.radians(self.psi))
        d = 0.0
        while d < 6.0:
            d += 0.01
            if self._inside_obstacle(self.x + c * d, self.y + s * d, r=0.0):
                return d
        return float("inf")

    # ------------------------------------------------------------ мережа
    def _reply(self, addr, text):
        if self.rng.random() < self.drop_ack:
            return
        self.sock.sendto(text.encode("ascii"), addr)

    def _cmd_loop(self):
        while self._run:
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            text = data.decode("ascii", "replace").strip()
            if self.rng.random() < self.drop_cmd:
                continue
            if text == "command":
                self.client_ip = addr[0]
            resp = self._immediate(text)
            if resp is not None:
                self.events.append((time.time(), text, resp))
                is_ext = text.startswith("EXT")
                if is_ext and self.rng.random() < self.ext_noise_before:
                    self._reply(addr, "unknown command: keepalive")
                    time.sleep(0.002)
                if resp != "":
                    self._reply(addr, resp)
                if is_ext and self.rng.random() < self.ext_noise_after:
                    time.sleep(0.002)
                    self._reply(addr, "unknown command: keepalive")
                continue
            if self.busy and text == "land":        # посадка перериває рух
                self.interrupt.set()
                t_end = time.time() + 2.0
                while self.busy and time.time() < t_end:
                    time.sleep(0.01)
            if self.busy:
                self.events.append((time.time(), text, "error Not joystick"))
                self._reply(addr, "error Not joystick")
                continue
            self.busy = True
            threading.Thread(target=self._action, args=(text, addr), daemon=True).start()

    def _immediate(self, text):
        """Відповідь без затримки або None (команда руху -- окремим потоком)."""
        parts = text.split()
        head = parts[0] if parts else ""
        if text in ("command", "keepalive", "motoron", "motoroff"):
            return "ok"
        if text == "emergency":
            self.interrupt.set()
            self.flying, self.z = False, 0.0
            return "ok"
        if text == "stop":
            self.interrupt.set()
            return "ok"
        if text == "battery?":
            return str(int(self.battery))
        if text == "sn?":
            return self.sn
        if text == "sdk?":
            return "30"
        if text == "hardware?":
            return "RMTT"
        if text == "wifi?":
            return "90"
        if text == "speed?":
            return f"{self.speed:.1f}"
        if text == "time?":
            return f"{0 if self.t_takeoff is None else int(time.time() - self.t_takeoff)}s"
        if text == "EXT tof?":
            d = self.front_distance()
            mm = 8192 if d > self.tof_range_m else int(d * 1000 + self.rng.gauss(0, 8))
            return f"tof {mm}"
        if head == "EXT":
            if len(parts) > 1 and parts[1] == "led":
                return "led ok"
            if len(parts) > 1 and parts[1] == "mled":
                return "matrix ok"
            return "error"
        if text == "streamon":
            if not self.streaming:
                self.streaming = True
                threading.Thread(target=self._video_loop, daemon=True).start()
            return "ok"
        if text == "streamoff":
            self.streaming = False
            return "ok"
        if head in ("setbitrate", "setfps", "setresolution", "ap", "wifi", "downvision"):
            return "ok"
        if head == "port" and len(parts) == 3:
            self.state_port, self.video_port = int(parts[1]), int(parts[2])
            return "ok"
        if head == "speed" and len(parts) == 2:
            v = float(parts[1])
            if 10 <= v <= 100:
                self.speed = v
                return "ok"
            return "out of range"
        if head == "rc":
            return ""
        if head in ("takeoff", "land", "up", "down", "forward", "back", "left", "right", "cw", "ccw"):
            return None
        return "error"

    def _sleep(self, seconds):
        """Пауза з можливістю переривання командою stop / emergency."""
        return self.interrupt.wait(seconds * self.time_scale)

    def _action(self, text, addr):
        self.interrupt.clear()
        try:
            resp = self._do(text)
        finally:
            self.busy = False
        self.events.append((time.time(), text, resp))
        moving = text.split()[0] in ("up", "down", "forward", "back", "left", "right", "cw", "ccw")
        if moving and self.late_ack_s > 0 and self.rng.random() < self.late_ack_prob:
            threading.Timer(self.late_ack_s, self._reply, args=(addr, resp)).start()
            return
        self._reply(addr, resp)

    def _do(self, text):
        parts = text.split()
        head = parts[0]
        if head == "takeoff":
            if self.flying:
                return "error"
            if self.battery < 10:
                return "error No enough battery"
            self._sleep(1.5)
            self.flying, self.z, self.t_takeoff = True, 0.8, time.time()
            return "ok"
        if head == "land":
            if not self.flying:
                return "error Not flying"
            self.interrupt.clear()
            self._sleep(1.5)
            self.flying, self.z = False, 0.0
            return "ok"
        if not self.flying:
            return "error Not flying"
        val = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else -1
        if head in ("cw", "ccw"):
            if not 1 <= val <= 360:
                return "out of range"
            steps = max(1, val // 10)
            for _ in range(steps):
                if self._sleep(val / 90.0 / steps):
                    return "ok"
                self.psi += (-1 if head == "cw" else 1) * val / steps * self.yaw_scale
            return "ok"
        if not 20 <= val <= 500:
            return "out of range"
        if head in ("up", "down"):
            dz = (val if head == "up" else -val) / 100.0
            self.vel = (0.0, 0.0, math.copysign(self.speed / 10.0, dz))
            self._sleep(abs(dz) * 100.0 / self.speed)
            self.vel = (0.0, 0.0, 0.0)
            self.z = max(0.1, self.z + dz)
            return "ok"
        ang = {"forward": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}[head]
        c, s = math.cos(math.radians(self.psi + ang)), math.sin(math.radians(self.psi + ang))
        dist = val / 100.0
        steps = max(1, int(dist / 0.02))
        self.vel = (c * self.speed / 10.0, s * self.speed / 10.0, 0.0)
        try:
            for _ in range(steps):
                if self._sleep(dist * 100.0 / self.speed / steps):
                    return "ok"
                nx, ny = self.x + c * dist / steps, self.y + s * dist / steps
                if self.z < 1.9 and self._inside_obstacle(nx, ny):
                    self.crashed, self.flying, self.z = True, False, 0.0
                    return "error Motor stop"
                self.x, self.y = nx, ny
        finally:
            self.vel = (0.0, 0.0, 0.0)
        return "ok"

    def _state_loop(self):
        while self._run:
            time.sleep(0.1)
            if self.flying:
                self.battery = max(0.0, self.battery - 0.1 / 6.0)
                self.temp = min(95.0, self.temp + 0.02)
            if self.client_ip is None or self.rng.random() < self.drop_state:
                continue
            tof = int(self.z * 100 + self.rng.gauss(0, 1)) if self.flying else 10
            msg = (f"mid:-1;x:-100;y:-100;z:-100;mpry:0,0,0;pitch:0;roll:0;yaw:{int(round(self.yaw_tello))};"
                   f"vgx:{int(round(self.vel[0]))};vgy:{int(round(self.vel[1]))};vgz:{int(round(self.vel[2]))};templ:{int(self.temp)};temph:{int(self.temp) + 2};tof:{tof};"
                   f"h:{int(self.z * 100)};bat:{int(self.battery)};baro:{100 + self.z:.2f};"
                   f"time:{0 if self.t_takeoff is None else int(time.time() - self.t_takeoff)};"
                   f"agx:0.00;agy:0.00;agz:-1000.00;\r\n")
            try:
                self.out.sendto(msg.encode("ascii"), (self.client_ip, self.state_port))
            except OSError:
                pass

    def _video_loop(self):
        clip = get_clip()
        if not clip:
            return
        t_next = time.time()
        k = 0
        while self.streaming and self._run:
            idx, au = clip[k % len(clip)]
            for off in range(0, len(au), PKT):
                if self.rng.random() >= self.drop_video:
                    try:
                        self.out.sendto(au[off:off + PKT], (self.client_ip, self.video_port))
                    except OSError:
                        pass
            self.video_log.append((time.time(), idx))
            k += 1
            t_next += 1.0 / 30.0
            time.sleep(max(0.0, t_next - time.time()))

    def close(self):
        self._run = False
        self.streaming = False
        self.interrupt.set()
        self.sock.close()
        self.out.close()


def start_fleet(n=3, base="127.0.0.11", room=None, starts=None, **kw):
    """starts -- список (x, y) або (x, y, курс) для кожного дрона; типово (1, 1), курс 0."""
    prefix, last = base.rsplit(".", 1)
    drones = []
    for k in range(n):
        st = (starts[k] if starts else (1.0, 1.0))
        heading = st[2] if len(st) > 2 else 0.0
        drones.append(FakeDrone(f"{prefix}.{int(last) + k}", f"0TQZSIM{k + 1:04d}", room=room,
                                start=st[:2], heading_deg=heading, seed=k, **kw))
    return drones


def run_emulator(n=3, base="127.0.0.11", room=None):
    get_clip()
    drones = start_fleet(n, base, room)
    print(f"Емулятор: {n} дронів на {drones[0].ip} ... {drones[-1].ip}, порт 8889; "
          f"відео: {'є' if get_clip() else 'немає (ffmpeg не знайдено)'}")
    print("В іншому вікні: python app.py scan --hosts " + f"{base}-{int(base.rsplit('.', 1)[1]) + n - 1}")
    print("Ctrl+C -- зупинити")
    try:
        while True:
            time.sleep(2.0)
            print(" | ".join(f"{d.ip}: {'у повітрі' if d.flying else 'на землі'}"
                             f"{' АВАРІЯ' if d.crashed else ''} ({d.x:.2f}, {d.y:.2f}, {d.z:.2f}) "
                             f"курс {d.psi:.0f}°" for d in drones))
    except KeyboardInterrupt:
        pass
    for d in drones:
        d.close()
