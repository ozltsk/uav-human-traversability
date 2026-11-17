"""Склад флоту (fleet.json) і пошук дронів у мережі роутера.

fleet.json зберігає відповідність «номер на наліпці -- серійний номер -- IP»:
  {"drones": [{"label": "1", "sn": "0TQZ...", "ip": "192.168.0.101"}, ...]}
Номер на наліпці ставиться один раз; IP оновлюється кожним `scan`.
"""
import ipaddress
import json
import os
import socket
import time

from drone import Drone, DroneError

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_PATH = os.path.join(HERE, "fleet.json")
AP_IP = "192.168.10.1"          # адреса дрона в режимі прямого підключення


def load_fleet(path=FLEET_PATH):
    if not os.path.exists(path):
        return {"drones": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fleet(fleet, path=FLEET_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fleet, f, ensure_ascii=False, indent=2)


def local_networks():
    """Мережі /24 усіх IPv4-адрес цього комп'ютера, крім 127.*."""
    nets = []
    try:
        addrs = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        addrs = []
    for a in addrs:
        if not a.startswith("127."):
            nets.append(str(ipaddress.ip_network(a + "/24", strict=False)))
    return sorted(set(nets))


def parse_hosts(spec):
    """"192.168.0.0/24", "127.0.0.11-15", "192.168.0.5,192.168.0.7" -> список IP."""
    hosts = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "/" in part:
            hosts.extend(str(h) for h in ipaddress.ip_network(part, strict=False).hosts())
        elif "-" in part.rsplit(".", 1)[-1]:
            base, rng = part.rsplit(".", 1)
            lo, hi = (int(v) for v in rng.split("-"))
            hosts.extend(f"{base}.{k}" for k in range(lo, hi + 1))
        else:
            hosts.append(part)
    return hosts


def probe(link, hosts, wait_s=2.5):
    """Надсилає "command" усім адресам; повертає IP, що відповіли "ok"."""
    while not link.cmd.unknown.empty():
        link.cmd.unknown.get_nowait()
    for k, ip in enumerate(hosts):
        try:
            link.cmd.send(ip, "command")
        except OSError:
            pass
        if k % 32 == 31:
            time.sleep(0.01)
    found = {}
    t_end = time.time() + wait_s
    while time.time() < t_end:
        try:
            _t, ip, text = link.cmd.unknown.get(timeout=0.1)
        except Exception:
            continue
        if text.lower().startswith("ok"):
            found[ip] = True
    return sorted(found, key=lambda s: tuple(int(x) for x in s.split(".")))


def discover(link, hosts, fleet):
    """Пошук дронів, зіставлення за серійним номером, оновлення fleet.json."""
    ips = probe(link, hosts)
    by_sn = {d["sn"]: d for d in fleet["drones"] if d.get("sn")}
    used = {d["label"] for d in fleet["drones"]}
    found = []
    for ip in ips:
        d = Drone(link, ip)
        info = d.read_info()
        sn = info.get("sn") or f"unknown-{ip}"
        entry = by_sn.get(sn)
        if entry is None:
            label = next(str(k) for k in range(1, 100) if str(k) not in used)
            used.add(label)
            entry = {"label": label, "sn": sn}
            fleet["drones"].append(entry)
            by_sn[sn] = entry
        entry["ip"] = ip
        entry["hardware"] = info.get("hardware")
        d.label = entry["label"]
        found.append((entry, info))
    fleet["drones"].sort(key=lambda e: int(e["label"]) if e["label"].isdigit() else 999)
    return found


def fleet_drones(link, fleet, labels, ap=False, ap_label="1"):
    """Об'єкти Drone для вибраних номерів; кожному надсилається "command".
    ap=True -- один дрон у прямому підключенні; ap_label -- номер на його наліпці."""
    if ap:
        entries = [{"label": str(ap_label), "sn": "", "ip": AP_IP}]
    else:
        entries = fleet["drones"] if labels in (None, "all") else \
            [e for e in fleet["drones"] if e["label"] in labels.split(",")]
        missing = set() if labels in (None, "all") else set(labels.split(",")) - {e["label"] for e in entries}
        if missing:
            raise SystemExit(f"У fleet.json немає дронів {sorted(missing)}: спершу `python app.py scan`")
    drones = []
    for e in entries:
        d = Drone(link, e["ip"], e["label"], e.get("sn", ""))
        try:
            d.connect()
        except DroneError as exc:
            print(f"Дрон {e['label']} ({e['ip']}) не відповідає: {exc} -- пропущено")
            continue
        drones.append(d)
    if not drones:
        raise SystemExit("Жоден дрон не відповів.")
    return drones


def setup_router(link, ssid, password, ip=AP_IP):
    """Перемикає дрон, до якого комп'ютер підключено напряму, на роботу через роутер."""
    for name, val in (("SSID", ssid), ("пароль", password)):
        if not val or " " in val or not val.isascii():
            raise SystemExit(f"{name} має бути латиницею без пробілів")
    d = Drone(link, ip, "AP")
    d.connect()
    info = d.read_info()
    print(f"Дрон {info.get('sn')} ({info.get('hardware')}), SDK {info.get('sdk')}, заряд {info.get('battery')} %")
    try:
        resp = d.send(f"ap {ssid} {password}", timeout=5.0, expect_ok=False)
    except DroneError:
        resp = None          # дрон міг перезавантажитися, не встигнувши відповісти
    print(f"Відповідь на ap: {resp or 'немає (нормально, якщо дрон уже перезавантажується)'}")
    return info, resp
