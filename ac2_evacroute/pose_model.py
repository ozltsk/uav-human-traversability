"""Правилова модель прохідності для людини в трьох позах.

Поза характеризується потрібною висотою H, шириною тіла W і швидкістю v.
Клітинка допустима для пози, якщо просвіт над опорою не менший за H + mu
(mu -- запас безпеки) і частка невідомих вокселів у смузі пози не більша
за u_max; ширина проходу враховується перетворенням відстані: відстань до
найближчої недопустимої клітинки має перевищувати W / 2 + mu. Стан клітинки
-- найшвидша допустима поза; якщо жодна не допустима, клітинка непрохідна,
а за великої частки невідомих вокселів -- недостатньо оглянута.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

# назва, висота H, ширина W, швидкість v (ISO 7250-1 із запасом на спорядження)
POSES = (("walk", 1.75, 0.5, 1.2), ("crouch", 1.2, 0.5, 0.5), ("crawl", 0.5, 0.6, 0.25))
WALK, CROUCH, CRAWL, BLOCKED, UNSEEN = 0, 1, 2, 3, 4
STATE_NAMES = ("прохідна пішки", "лише пригнувшись", "лише повзком", "непрохідна", "недостатньо оглянута")
SPEEDS = np.array([1.2, 0.5, 0.25, 0.0, 0.0])
V_CLIMB = 0.3
MAX_STEP = 0.4


def safety_margin(eps, mu0=1.0, range_m=2.0):
    """Запас безпеки, пропорційний самооцінці похибки глибини та дальності."""
    return float(mu0 * eps * range_m)


def width_check(admissible, k, voxel=0.1, mu=0.0):
    """Урахування ширини тіла: клітинка придатна для пози k, якщо відстань до
    найближчої недопустимої клітинки більша за половину ширини із запасом."""
    _, _, W, _ = POSES[k]
    dist = distance_transform_edt(admissible) * voxel
    return admissible & (dist > W / 2.0 + mu)


def rule_state_map(layers, voxel=0.1, mu=0.0, u_max=1.0, u_unseen=0.9, check_width=True):
    """Карта станів клітинок за правиловою моделлю поз. За check_width=False
    повертається стан колонки без урахування ширини проходу (використовується
    для міток нейромережі; ширину тоді враховує післяобробка її рішень)."""
    support, clearance = layers["support"], layers["clearance"]
    state = np.full(support.shape, BLOCKED, dtype=np.int8)
    for k in (CRAWL, CROUCH, WALK):            # від найповільнішої до найшвидшої
        name, H, W, _ = POSES[k]
        adm = (clearance >= H + mu) & (layers["unknown_" + name] <= u_max)
        if check_width:
            adm = width_check(adm, k, voxel, mu)
        state[adm] = k
    unseen = (state == BLOCKED) & (layers["unknown_crawl"] > u_unseen)
    state[unseen] = UNSEEN
    return state


def truth_state_map(truth_occ, voxel=0.1, check_width=True):
    """Точна карта станів за точною сіткою зайнятості (без невідомих вокселів)."""
    from map_layers import column_layers
    layers = column_layers(truth_occ.astype(np.int8) * 2, voxel)
    return rule_state_map(layers, voxel, mu=0.0, u_max=1.0, u_unseen=2.0, check_width=check_width), layers


def label_map(truth_state, obs_layers, u_unseen=0.9):
    """Мітки для навчання нейромережі: точний стан клітинки, а для колонок,
    яких дрон практично не бачив, -- клас "недостатньо оглянута"."""
    y = truth_state.copy()
    y[obs_layers["unknown_crawl"] > u_unseen] = UNSEEN
    return y
