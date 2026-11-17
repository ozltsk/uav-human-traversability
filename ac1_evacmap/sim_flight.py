"""Синтетичний обліт зруйнованого приміщення для перевірки програми без дрона.

Процедурний генератор будує приміщення 12 x 12 x 3 м, поділене двома
перегородками на три зони з прорізами випадкового типу (двері, вузька
щілина, низький проріз під просілою плитою, завалений проріз), купами
уламків, балками та нахиленими плитами. Карти глибини обчислюються
трасуванням променів з камери з полем зору 70 градусів. Похибку нейромережі
моделює перетворення: глибина множиться на гладке випадкове поле та
некорельований логнормальний шум, обернена глибина згладжується фільтром
Гаусса, після чого застосовується випадкове афінне перетворення; маска
підлоги містить 2 % хибних пікселів. Результат записується як журнал
обльоту FlightLog разом з точною сіткою зайнятості для перевірки карти.
"""
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from flight_log import FlightLog, FrameRecord
from mono_depth import CameraModel

# рівень похибки -> (відносний шум sigma, розмиття оберненої глибини, пікс.)
ERROR_LEVELS = {0: (0.0, 0.0), 1: (0.04, 0.6), 2: (0.08, 0.9), 3: (0.12, 1.2), 4: (0.16, 1.5)}
HIT_NONE, HIT_FLOOR, HIT_CEILING, HIT_BOX = 0, 1, 2, 3


# ---------------------------------------------------------------- сцена
def _box(x0, y0, z0, x1, y1, z1, kind):
    return {"lo": [float(x0), float(y0), float(z0)], "hi": [float(x1), float(y1), float(z1)], "kind": kind}


def _partition(x, rng, size, boxes, thickness=0.2):
    """Перегородка вздовж осі y на координаті x з 2-3 прорізами."""
    n_open = int(rng.integers(2, 4))
    centers = np.sort(rng.uniform(1.0, size[1] - 1.0, n_open))
    kinds = rng.choice(["door", "slit", "low", "blocked"], size=n_open, p=[0.35, 0.2, 0.25, 0.2])
    segments = []
    y_prev = 0.0
    for c, kind in zip(centers, kinds):
        if kind == "door":
            w, h = rng.uniform(0.9, 1.2), size[2]
        elif kind == "slit":
            w, h = rng.uniform(0.25, 0.4), size[2]
        elif kind == "low":
            w, h = rng.uniform(0.9, 1.2), rng.uniform(0.6, 1.1)
        else:
            w, h = rng.uniform(0.9, 1.2), size[2]
        y0, y1 = max(y_prev, c - w / 2), min(size[1], c + w / 2)
        if y0 - y_prev > 0.05:
            boxes.append(_box(x, y_prev, 0, x + thickness, y0, size[2], "wall"))
        if kind == "low":
            boxes.append(_box(x, y0, h, x + thickness, y1, size[2], "slab"))
        if kind == "blocked":
            boxes.append(_box(x - 0.4, y0, 0, x + thickness + 0.4, y1, rng.uniform(0.9, 1.6), "rubble"))
        segments.append({"x": x, "y0": y0, "y1": y1, "kind": str(kind), "height": h})
        y_prev = y1
    if size[1] - y_prev > 0.05:
        boxes.append(_box(x, y_prev, 0, x + thickness, size[1], size[2], "wall"))
    return segments


def make_scene(seed, size=(12.0, 12.0, 3.0)):
    """Процедурна сцена: словник з боксами, зонами, положенням постраждалого та виходом."""
    rng = np.random.default_rng(seed)
    boxes = []
    t = 0.3
    boxes.append(_box(-t, -t, 0, 0, size[1] + t, size[2], "wall"))
    boxes.append(_box(size[0], -t, 0, size[0] + t, size[1] + t, size[2], "wall"))
    boxes.append(_box(0, -t, 0, size[0], 0, size[2], "wall"))
    boxes.append(_box(0, size[1], 0, size[0], size[1] + t, size[2], "wall"))
    openings = []
    for x in (4.0, 8.0):
        openings.extend(_partition(x, rng, size, boxes))
    zones = [(0.3, 3.8), (4.4, 7.8), (8.4, 11.7)]
    for zx0, zx1 in zones:
        for _ in range(int(rng.integers(3, 6))):           # купи уламків
            w, d = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)
            x0, y0 = rng.uniform(zx0, zx1 - w), rng.uniform(0.3, size[1] - 0.3 - d)
            boxes.append(_box(x0, y0, 0, x0 + w, y0 + d, rng.uniform(0.2, 1.3), "rubble"))
        for _ in range(int(rng.integers(1, 3))):           # балки
            L = rng.uniform(2.0, 4.0)
            h = rng.uniform(0.5, 1.3)
            if rng.random() < 0.5:
                x0, y0 = rng.uniform(zx0, zx1 - 0.15), rng.uniform(0.3, size[1] - L)
                boxes.append(_box(x0, y0, h, x0 + 0.15, y0 + L, h + 0.15, "beam"))
            else:
                x0, y0 = rng.uniform(zx0, max(zx0 + 0.1, zx1 - L)), rng.uniform(0.3, size[1] - 0.3)
                boxes.append(_box(x0, y0, h, min(x0 + L, zx1), y0 + 0.15, h + 0.15, "beam"))
        if rng.random() < 0.6:                             # нахилена плита: східці з боксів
            w = rng.uniform(1.5, 2.5)
            x0, y0 = rng.uniform(zx0, zx1 - 1.2), rng.uniform(0.3, size[1] - 0.3 - w)
            steps = 4
            for k in range(steps):
                z = 0.2 + 1.0 * k / (steps - 1)
                boxes.append(_box(x0 + 0.3 * k, y0, z - 0.1, x0 + 0.3 * (k + 1), y0 + w, z + 0.05, "slab"))
    victim = [float(rng.uniform(9.0, 11.4)), float(rng.uniform(1.0, size[1] - 1.0))]
    exit_cell = [0.5, float(rng.uniform(2.0, size[1] - 2.0))]
    boxes = [b for b in boxes if not (b["kind"] == "rubble" and
                                      abs(b["lo"][0] - victim[0]) < 1.0 and abs(b["lo"][1] - victim[1]) < 1.0)]
    return {"seed": int(seed), "size": list(size), "boxes": boxes, "openings": openings,
            "zones": zones, "victim": victim, "exit": exit_cell}


def voxelize(scene, voxel=0.1):
    """Точна сітка зайнятості сцени у вокселах карти."""
    size = scene["size"]
    shape = tuple(int(round(s / voxel)) for s in size)
    occ = np.zeros(shape, dtype=bool)
    for b in scene["boxes"]:
        lo = np.maximum(np.floor(np.asarray(b["lo"]) / voxel).astype(int), 0)
        hi = np.minimum(np.ceil(np.asarray(b["hi"]) / voxel).astype(int), shape)
        if np.all(hi > lo):
            occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return occ


# ---------------------------------------------------------------- трасування
def ray_trace(origin, dirs, scene):
    """Перетин променів origin + t * dirs зі сценою. Для напрямків, нормованих
    так, що компонента вздовж оптичної осі дорівнює 1, параметр t дорівнює
    глибині Z. Повертає t, тип перетину та нормаль поверхні."""
    n = dirs.shape[0]
    size = scene["size"]
    t_best = np.full(n, np.inf)
    kind = np.full(n, HIT_NONE, dtype=np.int8)
    normal = np.zeros((n, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = dirs[:, 2]
        tf = np.where(dz < 0, -origin[2] / dz, np.inf)
        hit = tf < t_best
        t_best[hit], kind[hit], normal[hit] = tf[hit], HIT_FLOOR, (0, 0, 1)
        tc = np.where(dz > 0, (size[2] - origin[2]) / dz, np.inf)
        hit = tc < t_best
        t_best[hit], kind[hit], normal[hit] = tc[hit], HIT_CEILING, (0, 0, -1)
        inv = 1.0 / dirs
        for b in scene["boxes"]:
            lo, hi = np.asarray(b["lo"]), np.asarray(b["hi"])
            t1 = (lo[None, :] - origin[None, :]) * inv
            t2 = (hi[None, :] - origin[None, :]) * inv
            tnear, tfar = np.minimum(t1, t2), np.maximum(t1, t2)
            t_in = tnear.max(axis=1)
            t_out = tfar.min(axis=1)
            axis = tnear.argmax(axis=1)
            hit = (t_out >= t_in) & (t_in > 1e-6) & (t_in < t_best)
            if hit.any():
                t_best[hit], kind[hit] = t_in[hit], HIT_BOX
                nrm = np.zeros((int(hit.sum()), 3))
                sign = -np.sign(dirs[hit, :][np.arange(int(hit.sum())), axis[hit]])
                nrm[np.arange(int(hit.sum())), axis[hit]] = sign
                normal[hit] = nrm
    return t_best, kind, normal


def render_frame(cam, scene, position, yaw_deg):
    """Точна глибина, маска підлоги та відтінене зображення кадру."""
    d_c = cam.pixel_dirs()
    R = cam.rotation_world(yaw_deg)
    d_w = (d_c @ R.T).reshape(-1, 3)
    Z, kind, normal = ray_trace(np.asarray(position, dtype=np.float64), d_w, scene)
    Z = Z.reshape(cam.height, cam.width)
    kind = kind.reshape(cam.height, cam.width)
    light = np.array([0.4, 0.3, 0.87])
    shade = 0.35 + 0.65 * np.abs(normal @ light)
    fog = np.exp(-np.clip(Z.reshape(-1), 0, 15) / 9.0)
    gray = np.clip(255 * shade * (0.45 + 0.55 * fog), 0, 255).reshape(cam.height, cam.width)
    gray[kind == HIT_FLOOR] *= 0.85
    image = np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2)
    Z[kind == HIT_NONE] = np.inf
    return Z, kind == HIT_FLOOR, image


def emulate_network(Z, floor_mask, level, rng):
    """Перетворення точної глибини на вихід нейромережі з похибкою рівня level."""
    sigma, blur = ERROR_LEVELS[int(level)]
    H, W = Z.shape
    Zc = np.clip(np.where(np.isfinite(Z), Z, 15.0), 0.1, 15.0)
    if sigma > 0:
        coarse = rng.normal(0.0, sigma, size=(6, 8))
        field = zoom(coarse, (H / 6.0, W / 8.0), order=3)[:H, :W]
        noise = rng.normal(0.0, sigma / 2.0, size=Z.shape)
        Zn = Zc * np.exp(field) * np.exp(noise)
    else:
        Zn = Zc
    inv = 1.0 / Zn
    if blur > 0:
        inv = gaussian_filter(inv, blur)
    a, b = rng.uniform(0.6, 2.5), rng.uniform(0.0, 0.5)
    rel = (a * inv + b).astype(np.float32)
    mask = floor_mask.copy()
    flip = rng.random(size=mask.shape) < 0.02
    mask[flip] = ~mask[flip]
    return rel, mask


# ---------------------------------------------------------------- обліт
def _clear_of_boxes(scene, x, y, radius=0.45, z_max=2.0):
    for b in scene["boxes"]:
        if b["kind"] == "wall":
            continue
        lo, hi = b["lo"], b["hi"]
        if lo[2] < z_max and lo[0] - radius < x < hi[0] + radius and lo[1] - radius < y < hi[1] + radius:
            return False
    return True


def plan_waypoints(scene, rng, per_zone=4, tries=200):
    """Точки зависання над чистою підлогою: по per_zone у кожній зоні."""
    pts = []
    for zx0, zx1 in scene["zones"]:
        got = 0
        for _ in range(tries):
            x, y = rng.uniform(zx0 + 0.6, zx1 - 0.6), rng.uniform(1.0, scene["size"][1] - 1.0)
            if _clear_of_boxes(scene, x, y) and all(np.hypot(x - px, y - py) > 1.5 for px, py in pts):
                pts.append((float(x), float(y)))
                got += 1
                if got >= per_zone:
                    break
    return pts


def simulate_flight(out_dir, seed=0, level=2, width=320, height=240, turns=4,
                    height_range=(1.0, 1.6), per_zone=4, save_frames=True):
    rng = np.random.default_rng(10_000 + seed)
    scene = make_scene(seed)
    cam = CameraModel(width, height)
    waypoints = plan_waypoints(scene, rng, per_zone=per_zone)
    log = FlightLog(root=out_dir)
    log.prepare()
    log.meta = {"source": "simulator", "seed": int(seed), "error_level": int(level),
                "frame_size": [width, height], "hfov_deg": cam.hfov_deg, "tilt_deg": cam.tilt_deg,
                "room_size": scene["size"], "waypoints": [list(p) for p in waypoints]}
    index = 0
    for wi, (x, y) in enumerate(waypoints):
        h = float(rng.uniform(*height_range))
        for k in range(turns):
            yaw = 90.0 * k + float(rng.normal(0.0, 1.5))
            Z, floor_mask, image = render_frame(cam, scene, (x, y, h), yaw)
            rel, mask = emulate_network(Z, floor_mask, level, rng)
            rec = FrameRecord(index=index, x=x, y=y, z_tof=h + float(rng.normal(0.0, 0.01)),
                              yaw_deg=yaw, waypoint=wi)
            log.add_frame(rec, image=image if save_frames else None, rel_inv_depth=rel, floor_mask=mask)
            index += 1
    log.save()
    with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(out_dir, "truth_occupancy.npz"), occupancy=voxelize(scene))
    return log, scene
