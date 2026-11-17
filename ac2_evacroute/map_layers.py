"""Шари карти: з 3D-карти станів (вільно / невідомо / зайнято) для кожної
колонки обчислюються висота опори, вільний просвіт, частки невідомих
вокселів у смугах поз людини та частка оглянутих вокселів. Ці шари є
входом правилової моделі прохідності та нейромережі прохідності.
"""
import numpy as np

FREE, UNKNOWN, OCCUPIED = 0, 1, 2
POSE_NAMES = ("walk", "crouch", "crawl")
POSE_HEIGHT = {"walk": 1.75, "crouch": 1.2, "crawl": 0.5}
FEATURE_NAMES = ("support_height", "clearance", "unknown_walk", "unknown_crouch",
                 "unknown_crawl", "occupied_density", "observed_fraction")


def load_map_npz(path):
    """Карта з файлу першої програми (логарифм відношення шансів) або
    збережена карта станів. Повертає (state, voxel, eps_median)."""
    z = np.load(path, allow_pickle=False)
    voxel = float(z["voxel"]) if "voxel" in z else 0.1
    if "state" in z:
        state = z["state"].astype(np.int8)
    else:
        lo, hi = float(z["l_lo"]), float(z["l_hi"])
        L = z["logodds"]
        state = np.full(L.shape, UNKNOWN, dtype=np.int8)
        state[L > hi] = OCCUPIED
        state[L < lo] = FREE
    eps = float(np.median(z["eps_history"])) if "eps_history" in z and z["eps_history"].size else 0.0
    return state, voxel, eps


def column_layers(state, voxel=0.1, analysis_height=2.2):
    """Повертає словник шарів розміром (nx, ny):
    support  -- висота опори: межа першого незайнятого воксела над підлогою/уламками;
    clearance -- вільний просвіт над опорою до першого зайнятого воксела;
    unknown_<pose> -- частка невідомих вокселів у смузі висотою пози над опорою;
    density  -- частка зайнятих вокселів у колонці до analysis_height;
    observed -- частка оглянутих (не невідомих) вокселів у колонці до analysis_height."""
    nx, ny, nz = state.shape
    occ = state == OCCUPIED
    unk = state == UNKNOWN
    free = state == FREE
    ks = np.arange(nz)[None, None, :]
    # опора: межа першого вільного воксела над блоком зайнятих/невідомих вокселів,
    # що починається від підлоги, якщо в цьому блоці є видима (зайнята) поверхня;
    # інакше (нутро не спостережене, поверхні не видно) опорою вважається підлога
    k_f = np.where(free.any(axis=2), free.argmax(axis=2), nz)
    block_has_surface = (occ & (ks < k_f[..., None])).any(axis=2)
    k_s = np.where(block_has_surface, k_f, 0)
    occ_above = occ & (ks >= k_s[..., None])
    k_c = np.where(occ_above.any(axis=2), occ_above.argmax(axis=2), nz)
    support = k_s * voxel
    clearance = (k_c - k_s) * voxel
    layers = {"support": support.astype(np.float32), "clearance": clearance.astype(np.float32)}
    for name in POSE_NAMES:
        n_band = max(1, int(round(POSE_HEIGHT[name] / voxel)))
        band = (ks >= k_s[..., None]) & (ks < (k_s + n_band)[..., None])
        n_in = band.sum(axis=2)
        layers["unknown_" + name] = np.where(n_in > 0, (unk & band).sum(axis=2) / np.maximum(n_in, 1), 1.0).astype(np.float32)
    k_a = min(nz, int(round(analysis_height / voxel)))
    layers["density"] = occ[:, :, :k_a].mean(axis=2).astype(np.float32)
    layers["observed"] = (~unk[:, :, :k_a]).mean(axis=2).astype(np.float32)
    return layers


def features_from_layers(layers, room_height=3.0):
    """Тензор ознак (7, nx, ny) для нейромережі прохідності."""
    f = np.stack([np.clip(layers["support"], 0, room_height) / room_height,
                  np.clip(layers["clearance"], 0, room_height) / room_height,
                  layers["unknown_walk"], layers["unknown_crouch"], layers["unknown_crawl"],
                  layers["density"], layers["observed"]], axis=0)
    return f.astype(np.float32)


def features_from_state(state, voxel=0.1):
    return features_from_layers(column_layers(state, voxel))
