"""Транспорт UDP для DJI RoboMaster TT / Tello (SDK 3.0) на весь флот.

Три сокети на всі дрони, кожен розбирає пакети за IP-адресою відправника:
  команди -- локальний порт 8889; дрон відповідає на адресу й порт відправника;
  стан    -- порт 8890; рядок "pitch:0;roll:0;...;" приблизно 10 разів на секунду;
  відео   -- порт 11111; сирий потік H.264 (Annex B), кадр ділиться на пакети <= 1460 байт.
Відео під час польоту не декодується: байти пишуться у файл разом із часом
надходження кожного пакета, декодування виконується після посадки
(postprocess.py). Так слабкий ноутбук не втрачає пакети навіть із п'ятьма потоками.
"""
import collections
import os
import queue
import socket
import threading
import time

VERSION = "2026-11-04 v3"      # версія софту; пишеться в meta.json кожного запису

CMD_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111
DRONE_CMD_PORT = 8889

# поля рядка стану SDK 3.0 у тому порядку, в якому вони пишуться в state.csv
STATE_KEYS = ["mid", "x", "y", "z", "mpry", "pitch", "roll", "yaw", "vgx", "vgy", "vgz",
              "templ", "temph", "tof", "h", "bat", "baro", "time", "agx", "agy", "agz"]


def parse_state(text):
    """Рядок стану -> словник; числові поля -> float, "mpry" лишається рядком."""
    out = {}
    for part in text.strip().split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        if key == "mpry":
            out[key] = value.strip()
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value.strip()
    return out


def _udp_socket(bind_ip, port, rcvbuf=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if rcvbuf:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    sock.bind((bind_ip, port))
    # Windows: без цього recvfrom падає з ConnectionResetError (10054), щойно
    # якийсь вузол відповів ICMP "порт недосяжний" (типово під час сканування мережі)
    if os.name == "nt" and hasattr(socket, "SIO_UDP_CONNRESET"):
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)
    sock.settimeout(0.2)
    return sock


class _Hub:
    """Спільна частина: сокет, потік приймання, лічильники за IP."""

    def __init__(self, port, bind_ip="0.0.0.0", rcvbuf=None):
        self.port = port
        self.sock = _udp_socket(bind_ip, port, rcvbuf)
        self.lock = threading.Lock()
        self.stats = {}                    # ip -> [пакетів, байтів, час останнього]
        self._run = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _count(self, ip, n, t):
        s = self.stats.get(ip)
        if s is None:
            s = self.stats[ip] = [0, 0, 0.0]
        s[0] += 1
        s[1] += n
        s[2] = t

    def _loop(self):
        while self._run:
            try:
                data, (ip, _port) = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                if not self._run:
                    break
                continue
            t = time.time()
            self._count(ip, len(data), t)
            self.handle(t, ip, data)

    def handle(self, t, ip, data):
        raise NotImplementedError

    def close(self):
        self._run = False
        self.thread.join(timeout=1.0)
        self.sock.close()


class CommandHub(_Hub):
    """Надсилання команд і розбирання відповідей за IP дрона."""

    def __init__(self, port=CMD_PORT, bind_ip="0.0.0.0"):
        self.queues = {}                   # ip -> Queue[(t, text)]
        self.unknown = queue.Queue()       # відповіді від незареєстрованих адрес (сканування)
        super().__init__(port, bind_ip)

    def register(self, ip):
        with self.lock:
            return self.queues.setdefault(ip, queue.Queue())

    def send(self, ip, text, port=DRONE_CMD_PORT):
        self.sock.sendto(text.encode("ascii"), (ip, port))

    def handle(self, t, ip, data):
        text = data.decode("utf-8", "replace").strip()
        q = self.queues.get(ip)
        if q is not None:
            q.put((t, text))
        else:
            self.unknown.put((t, ip, text))


class StateLog:
    """Запис рядків стану одного дрона у CSV."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", newline="")
        self.f.write("t," + ",".join(STATE_KEYS) + "\n")

    def write(self, t, st):
        vals = []
        for k in STATE_KEYS:
            v = st.get(k, "")
            # "mpry" має вигляд "0,0,0": коми замінюються пробілами, щоб не ламати CSV
            vals.append(v.replace(",", " ") if isinstance(v, str) else f"{v:g}")
        self.f.write(f"{t:.4f}," + ",".join(vals) + "\n")

    def close(self):
        self.f.close()


class StateHub(_Hub):
    """Останній стан кожного дрона + необов'язковий запис у файл."""

    def __init__(self, port=STATE_PORT, bind_ip="0.0.0.0"):
        self.latest = {}                   # ip -> (t, dict)
        self.history = {}                  # ip -> deque[(t, dict)] за останні ~60 с
        self.logs = {}                     # ip -> StateLog
        super().__init__(port, bind_ip)

    def handle(self, t, ip, data):
        st = parse_state(data.decode("ascii", "replace"))
        if not st:
            return
        self.latest[ip] = (t, st)
        hist = self.history.get(ip)
        if hist is None:
            hist = self.history[ip] = collections.deque(maxlen=600)
        hist.append((t, st))
        with self.lock:
            log = self.logs.get(ip)
            if log is not None:
                log.write(t, st)

    def attach(self, ip, log):
        with self.lock:
            self.logs[ip] = log

    def detach(self, ip):
        with self.lock:
            log = self.logs.pop(ip, None)
        if log is not None:
            log.close()

    def get(self, ip):
        """(вік стану в секундах, словник) або (None, {})."""
        item = self.latest.get(ip)
        if item is None:
            return None, {}
        return time.time() - item[0], item[1]


class VideoRecorder:
    """Сирий H.264 одного дрона + індекс пакетів (час, зсув, довжина)."""

    def __init__(self, path_h264, path_index):
        self.f = open(path_h264, "wb", buffering=1 << 20)
        self.idx = open(path_index, "w", encoding="utf-8", newline="", buffering=1 << 16)
        self.idx.write("t,offset,length\n")
        self.offset = 0
        self.packets = 0

    def write(self, t, data):
        self.f.write(data)
        self.idx.write(f"{t:.6f},{self.offset},{len(data)}\n")
        self.offset += len(data)
        self.packets += 1

    def close(self):
        self.f.close()
        self.idx.close()


class VideoHub(_Hub):
    """Приймання відеопотоків усіх дронів одним сокетом, розкладання за IP."""

    def __init__(self, port=VIDEO_PORT, bind_ip="0.0.0.0"):
        self.recorders = {}                # ip -> VideoRecorder
        super().__init__(port, bind_ip, rcvbuf=8 << 20)

    def handle(self, t, ip, data):
        with self.lock:
            rec = self.recorders.get(ip)
            if rec is not None:
                rec.write(t, data)

    def attach(self, ip, rec):
        with self.lock:
            self.recorders[ip] = rec

    def detach(self, ip):
        with self.lock:
            rec = self.recorders.pop(ip, None)
        if rec is not None:
            rec.close()


class Link:
    """Три концентратори разом; один екземпляр на процес."""

    def __init__(self, cmd_port=CMD_PORT, state_port=STATE_PORT, video_port=VIDEO_PORT,
                 bind_ip="0.0.0.0"):
        self.cmd = CommandHub(cmd_port, bind_ip)
        self.state = StateHub(state_port, bind_ip)
        self.video = VideoHub(video_port, bind_ip)

    def close(self):
        for hub in (self.cmd, self.state, self.video):
            hub.close()
