"""Генератор синтетичних сцен зруйнованих приміщень та модель спостереження.

Сцена: приміщення 12 x 12 x 3 м з двома перегородками, прорізами
випадкового типу (двері, вузька щілина, низький проріз, завалений проріз),
купами уламків, балками та нахиленими плитами. Для кожної сцени відомі
точна вокселна сітка зайнятості, положення постраждалого та вихід.

Модель спостереження імітує обліт дроном: з точок зависання над чистою
підлогою у всіх напрямках (чотири повороти камери) випускаються промені
обмеженої дальності; воксели вздовж променя стають вільними, воксел
перетину -- зайнятим, а відстань до перетину спотворюється похибкою,
пропорційною дальності (як у монокулярної нейромережевої глибини).
Результат -- спостережена карта станів: вільно / невідомо / зайнято,
з якої друга програма навчається оцінювати прохідність для людини.
"""
import numpy as np

VOXEL = 0.1
ROOM = (12.0, 12.0, 3.0)
FREE, UNKNOWN, OCCUPIED = 0, 1, 2
WALL_T = 0.3


def _box(x0, y0, z0, x1, y1, z1, kind):
    return {"lo": [float(x0), float(y0), float(z0)], "hi": [float(x1), float(y1), float(z1)], "kind": kind}


def _partition(x, rng, boxes, thickness=0.2):
    """Перегородка вздовж осі y на координаті x з 2-3 прорізами різного типу."""
    n_open = int(rng.integers(2, 4))
    centers = np.sort(rng.uniform(1.2, ROOM[1] - 1.2, n_open))
    kinds = rng.choice(["door", "slit", "low", "blocked"], size=n_open, p=[0.35, 0.2, 0.25, 0.2])
    y_prev = WALL_T
    openings = []
    for c, kind in zip(centers, kinds):
        if kind == "door":
            w, h = rng.uniform(0.9, 1.2), ROOM[2]
        elif kind == "slit":
            w, h = rng.uniform(0.25, 0.4), ROOM[2]
        elif kind == "low":
            w, h = rng.uniform(0.9, 1.2), rng.uniform(0.6, 1.1)
        else:
            w, h = rng.uniform(0.9, 1.2), ROOM[2]
        y0, y1 = max(y_prev, c - w / 2), min(ROOM[1] - WALL_T, c + w / 2)
        if y0 - y_prev > 0.05:
            boxes.append(_box(x, y_prev, 0, x + thickness, y0, ROOM[2], "wall"))
        if kind == "low":
            boxes.append(_box(x, y0, h, x + thickness, y1, ROOM[2], "slab"))
        if kind == "blocked":
            boxes.append(_box(x - 0.4, y0, 0, x + thickness + 0.4, y1, rng.uniform(0.9, 1.6), "rubble"))
        openings.append({"x": x, "y0": y0, "y1": y1, "kind": str(kind), "height": h})
        y_prev = y1
    if ROOM[1] - WALL_T - y_prev > 0.05:
        boxes.append(_box(x, y_prev, 0, x + thickness, ROOM[1] - WALL_T, ROOM[2], "wall"))
    return openings


def make_scene(seed):
    """Процедурна сцена; постраждалий у дальній зоні, вихід у стіні x = 0."""
    rng = np.random.default_rng(seed)
    boxes = []
    exit_y = float(rng.uniform(2.0, ROOM[1] - 2.0))
    # зовнішні стіни всередині сітки; у стіні x = 0 проріз виходу шириною 1 м
    boxes.append(_box(0, 0, 0, WALL_T, exit_y - 0.5, ROOM[2], "wall"))
    boxes.append(_box(0, exit_y + 0.5, 0, WALL_T, ROOM[1], ROOM[2], "wall"))
    boxes.append(_box(ROOM[0] - WALL_T, 0, 0, ROOM[0], ROOM[1], ROOM[2], "wall"))
    boxes.append(_box(0, 0, 0, ROOM[0], WALL_T, ROOM[2], "wall"))
    boxes.append(_box(0, ROOM[1] - WALL_T, 0, ROOM[0], ROOM[1], ROOM[2], "wall"))
    openings = []
    for x in (4.0, 8.0):
        openings.extend(_partition(x, rng, boxes))
    zones = [(0.5, 3.8), (4.4, 7.8), (8.4, 11.5)]
    for zx0, zx1 in zones:
        for _ in range(int(rng.integers(3, 6))):
            w, d = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)
            x0, y0 = rng.uniform(zx0, zx1 - w), rng.uniform(0.5, ROOM[1] - 0.5 - d)
            boxes.append(_box(x0, y0, 0, x0 + w, y0 + d, rng.uniform(0.2, 1.3), "rubble"))
        for _ in range(int(rng.integers(1, 3))):
            L, h = rng.uniform(2.0, 4.0), rng.uniform(0.5, 1.3)
            if rng.random() < 0.5:
                x0, y0 = rng.uniform(zx0, zx1 - 0.15), rng.uniform(0.5, ROOM[1] - 0.5 - L)
                boxes.append(_box(x0, y0, h, x0 + 0.15, y0 + L, h + 0.15, "beam"))
            else:
                x0, y0 = rng.uniform(zx0, max(zx0 + 0.1, zx1 - L)), rng.uniform(0.5, ROOM[1] - 0.5)
                boxes.append(_box(x0, y0, h, min(x0 + L, zx1), y0 + 0.15, h + 0.15, "beam"))
        if rng.random() < 0.6:
            w = rng.uniform(1.5, 2.5)
            x0, y0 = rng.uniform(zx0, zx1 - 1.2), rng.uniform(0.5, ROOM[1] - 0.5 - w)
            for k in range(4):
                z = 0.2 + 1.0 * k / 3.0
                boxes.append(_box(x0 + 0.3 * k, y0, z - 0.1, x0 + 0.3 * (k + 1), y0 + w, z + 0.05, "slab"))
    victim = [float(rng.uniform(9.0, 11.2)), float(rng.uniform(1.0, ROOM[1] - 1.0))]
    exit_xy = [0.6, exit_y]
    # чистий простір навколо постраждалого та перед виходом
    def near(b, p, r):
        return b["kind"] != "wall" and abs((b["lo"][0] + b["hi"][0]) / 2 - p[0]) < r and \
            abs((b["lo"][1] + b["hi"][1]) / 2 - p[1]) < r
    boxes = [b for b in boxes if not near(b, victim, 1.0) and not near(b, exit_xy, 1.2)]
    return {"seed": int(seed), "boxes": boxes, "openings": openings, "zones": zones,
            "victim": victim, "exit": exit_xy}


def voxelize(scene, voxel=VOXEL):
    shape = tuple(int(round(s / voxel)) for s in ROOM)
    occ = np.zeros(shape, dtype=bool)
    for b in scene["boxes"]:
        lo = np.maximum(np.floor(np.asarray(b["lo"]) / voxel).astype(int), 0)
        hi = np.minimum(np.ceil(np.asarray(b["hi"]) / voxel).astype(int), shape)
        if np.all(hi > lo):
            occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return occ


def hover_points(scene, rng, per_zone=4, tries=300, height_range=(1.0, 1.6)):
    """Точки зависання над чистою підлогою (по per_zone у кожній зоні)."""
    pts = []
    for zx0, zx1 in scene["zones"]:
        got = 0
        for _ in range(tries):
            x, y = rng.uniform(zx0 + 0.5, zx1 - 0.5), rng.uniform(1.0, ROOM[1] - 1.0)
            clear = True
            for b in scene["boxes"]:
                if b["kind"] == "wall" or b["lo"][2] > 2.0:
                    continue
                if b["lo"][0] - 0.45 < x < b["hi"][0] + 0.45 and b["lo"][1] - 0.45 < y < b["hi"][1] + 0.45:
                    clear = False
                    break
            if clear and all(np.hypot(x - px, y - py) > 1.5 for px, py, _ in pts):
                pts.append((float(x), float(y), float(rng.uniform(*height_range))))
                got += 1
                if got >= per_zone:
                    break
    return pts


def _directions(az_deg, el_deg):
    A, E = np.meshgrid(np.radians(az_deg), np.radians(el_deg), indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)


# Камера RoboMaster TT за натурними дослідами 02-04.11.2026: кут огляду 55,5° (калібрування 4 дронів),
# нахил -10° -> по вертикалі -31,5...+11,5°; 4 знімки через 90° у точці; висота за ToF 1,1-1,25 м;
# похибка положення точки ~0,3 м і курсу ~3° (числення шляху, мітки heading).
RMTT_CAMERA = {"el_range": (-31.5, 11.5), "az_sector": 55.5, "pose_sigma": 0.3, "yaw_sigma": 3.0,
               "height_range": (1.1, 1.25)}


def observe(occ, points, rng, rel_noise=0.06, frame_bias=0.03, max_range=4.0,
            az_step=1.0, el_range=(-40.0, 18.0), el_step=1.5, voxel=VOXEL,
            l_occ=0.85, l_free=-0.6, l_lo=-0.5, l_hi=1.0,
            az_sector=None, pose_sigma=0.0, yaw_sigma=0.0):
    """Спостережена карта станів за точною сіткою occ з точок зависання.
    az_sector -- None: огляд по колу (як було); інакше ширина кадру, град: 4 сектори через 90° від
    випадкового курсу, кожен зі своїм зсувом масштабу (frame_bias). pose_sigma (м), yaw_sigma (град) --
    похибка положення й курсу точки, з якою спостереження наносяться на карту (промені й перетини
    рахуються в справжній геометрії)."""
    shape = np.asarray(occ.shape)
    occ_flat = occ.reshape(-1)
    logodds = np.zeros(occ.size, dtype=np.float32)
    el_deg = np.arange(el_range[0], el_range[1] + 1e-9, el_step)
    steps = np.arange(voxel * 0.5, max_range, voxel * 0.7)
    S = steps.size
    dirs_full = _directions(np.arange(0.0, 360.0, az_step), el_deg) if az_sector is None else None
    for (x, y, h) in points:
        origin = np.array([x, y, h])
        if az_sector is None:
            dirs = dirs_full
            sector = np.zeros(dirs.shape[0], dtype=np.int64)
        else:
            base = rng.uniform(0.0, 360.0)
            half = np.arange(-az_sector / 2.0, az_sector / 2.0 + 1e-9, az_step)
            parts = [_directions(base + 90.0 * k + half, el_deg) for k in range(4)]
            dirs = np.vstack(parts)
            sector = np.repeat(np.arange(4), [p.shape[0] for p in parts])
        pts = origin[None, None, :] + dirs[:, None, :] * steps[None, :, None]
        idx = np.floor(pts / voxel).astype(np.int64)
        inside_true = np.all((idx >= 0) & (idx < shape), axis=-1)
        flat_true = np.ravel_multi_index(tuple(np.clip(idx[..., k], 0, shape[k] - 1) for k in range(3)), tuple(shape))
        hit = inside_true & occ_flat[flat_true]
        has_hit = hit.any(axis=1)
        first = np.where(has_hit, hit.argmax(axis=1), S)
        if pose_sigma > 0 or yaw_sigma > 0:             # нанесення на карту з хибним положенням і курсом
            dxy = rng.normal(0.0, pose_sigma, 2)
            th = np.radians(rng.normal(0.0, yaw_sigma))
            Rz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
            reg = (origin + np.array([dxy[0], dxy[1], 0.0]))[None, None, :] + (dirs @ Rz.T)[:, None, :] * steps[None, :, None]
            ridx = np.floor(reg / voxel).astype(np.int64)
            inside = np.all((ridx >= 0) & (ridx < shape), axis=-1)
            flat = np.ravel_multi_index(tuple(np.clip(ridx[..., k], 0, shape[k] - 1) for k in range(3)), tuple(shape))
        else:
            inside, flat = inside_true, flat_true
        # похибка дальності: зсув кадру + некорельований шум, пропорційні дальності
        bias = rng.normal(0.0, frame_bias, size=int(sector.max()) + 1)[sector]
        noise = 1.0 + bias + rng.normal(0.0, rel_noise, size=first.shape)
        perceived = np.clip(np.round(first * noise).astype(np.int64), 1, S)
        k = np.arange(S)[None, :]
        free_m = inside & (k < perceived[:, None] - 1)
        free_cells = np.unique(flat[free_m])
        logodds[free_cells] += l_free
        occ_m = has_hit & (perceived < S)
        occ_cells = flat[np.arange(dirs.shape[0])[occ_m], np.minimum(perceived[occ_m], S - 1)]
        occ_cells = occ_cells[inside[np.arange(dirs.shape[0])[occ_m], np.minimum(perceived[occ_m], S - 1)]]
        np.add.at(logodds, occ_cells, l_occ)
        np.clip(logodds, -4.0, 4.0, out=logodds)
    state = np.full(occ.size, UNKNOWN, dtype=np.int8)
    state[logodds > l_hi] = OCCUPIED
    state[logodds < l_lo] = FREE
    return state.reshape(occ.shape)
