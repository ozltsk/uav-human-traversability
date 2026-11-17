"""Імовірнісна 3D-карта зайнятості на основі логарифма відношення шансів.

Кожен воксел зберігає L = log(p / (1 - p)). Промінь пікселя звільняє
воксели, через які проходить (приріст l_free не більше одного разу за
кадр), а воксели поверхні товщиною s за точкою перетину отримують приріст
l_occ за кожен промінь. Воксел зайнятий, якщо L > l_hi, вільний, якщо
L < l_lo, інакше невідомий.
"""
import numpy as np

FREE, UNKNOWN, OCCUPIED = 0, 1, 2


class OccupancyMap:
    def __init__(self, size_xyz=(12.0, 12.0, 3.0), voxel=0.1, origin=(0.0, 0.0, 0.0),
                 l_occ=0.85, l_free=-0.6, l_min=-4.0, l_max=4.0, l_lo=-0.5, l_hi=1.0,
                 surface_thickness=0.2, max_range=4.0):
        self.voxel = float(voxel)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.shape = tuple(int(round(s / voxel)) for s in size_xyz)
        self.logodds = np.zeros(self.shape, dtype=np.float32)
        self.l_occ, self.l_free = l_occ, l_free
        self.l_min, self.l_max = l_min, l_max
        self.l_lo, self.l_hi = l_lo, l_hi
        self.surface_thickness = surface_thickness
        self.max_range = max_range
        self.frames_integrated = 0
        self.eps_history = []

    # ---------- координати ----------
    def to_index(self, pts):
        return np.floor((pts - self.origin) / self.voxel).astype(np.int64)

    def inside(self, idx):
        return np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=-1)

    def cell_center(self, idx):
        return self.origin + (np.asarray(idx) + 0.5) * self.voxel

    # ---------- інтегрування кадру ----------
    def integrate_points(self, cam_pos, points, eps=None):
        """Інтегрує хмару точок одного кадру, спостережену з точки cam_pos.
        Повертає кількість використаних точок."""
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        d = points - cam_pos[None, :]
        rng = np.linalg.norm(d, axis=1)
        ok = (rng > 0.15) & (rng < self.max_range)
        points, d, rng = points[ok], d[ok], rng[ok]
        if points.shape[0] == 0:
            return 0
        step = self.voxel * 0.7
        unit = d / rng[:, None]
        n_steps = np.ceil(rng / step).astype(np.int64)
        # вибірка точок уздовж усіх променів одразу: крок k для променя i дійсний,
        # якщо k < n_steps[i] - 1 (зупинка за один крок до поверхні)
        ks = np.arange(1, int(n_steps.max()))
        valid = ks[None, :] < (n_steps[:, None] - 1)
        p = cam_pos[None, None, :] + unit[:, None, :] * (ks[None, :, None] * step)
        idx = self.to_index(p[valid])
        idx = idx[self.inside(idx)]
        if idx.size:
            flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), self.shape)
            flat = np.unique(flat)
            self.logodds.reshape(-1)[flat] += self.l_free
        # поверхня товщиною s за точкою перетину: приріст за кожен промінь
        n_thick = max(1, int(round(self.surface_thickness / self.voxel)))
        occ_cells = []
        for k in range(n_thick):
            p = points + unit * (k * self.voxel)
            idx = self.to_index(p)
            idx = idx[self.inside(idx)]
            if idx.size:
                occ_cells.append(idx)
        if occ_cells:
            oc = np.concatenate(occ_cells, axis=0)
            np.add.at(self.logodds, (oc[:, 0], oc[:, 1], oc[:, 2]), self.l_occ)
        np.clip(self.logodds, self.l_min, self.l_max, out=self.logodds)
        self.frames_integrated += 1
        if eps is not None:
            self.eps_history.append(float(eps))
        return int(points.shape[0])

    # ---------- стан ----------
    def state(self):
        s = np.full(self.shape, UNKNOWN, dtype=np.int8)
        s[self.logodds > self.l_hi] = OCCUPIED
        s[self.logodds < self.l_lo] = FREE
        return s

    def probability(self):
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def summary(self):
        s = self.state()
        return {"voxels": int(s.size),
                "occupied": int((s == OCCUPIED).sum()),
                "free": int((s == FREE).sum()),
                "unknown": int((s == UNKNOWN).sum()),
                "frames": self.frames_integrated,
                "eps_median": float(np.median(self.eps_history)) if self.eps_history else None}

    def top_view(self, z_from=0.3, z_to=1.8):
        """Проєкція стану на площину: зайнято, якщо в діапазоні висот є
        зайнятий воксел; вільно, якщо всі воксели вільні; інакше невідомо."""
        s = self.state()
        k0, k1 = int(z_from / self.voxel), int(z_to / self.voxel)
        band = s[:, :, k0:k1]
        top = np.full(self.shape[:2], UNKNOWN, dtype=np.int8)
        top[(band == FREE).all(axis=2)] = FREE
        top[(band == OCCUPIED).any(axis=2)] = OCCUPIED
        return top

    def compare_with_truth(self, truth_occ, z_from=0.3, z_to=1.8):
        """Порівняння з точною сіткою зайнятості (та сама форма) у смузі висот:
        частка правильно класифікованих спостережених вокселів та кількість
        хибно вільних і хибно зайнятих."""
        s = self.state()
        k0, k1 = int(z_from / self.voxel), int(z_to / self.voxel)
        s, t = s[:, :, k0:k1], truth_occ[:, :, k0:k1].astype(bool)
        seen = s != UNKNOWN
        pred_occ = s == OCCUPIED
        return {"seen": int(seen.sum()),
                "accuracy": float(((pred_occ == t) & seen).sum() / max(1, seen.sum())),
                "false_free": int((seen & ~pred_occ & t).sum()),
                "false_occupied": int((seen & pred_occ & ~t).sum())}

    # ---------- збереження ----------
    def save(self, path, extra=None):
        meta = {"voxel": self.voxel, "origin": self.origin, "shape": np.asarray(self.shape),
                "l_lo": self.l_lo, "l_hi": self.l_hi, "frames": self.frames_integrated,
                "eps_history": np.asarray(self.eps_history, dtype=np.float32)}
        if extra:
            meta.update(extra)
        np.savez_compressed(path, logodds=self.logodds, **meta)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        shape = tuple(int(v) for v in z["shape"])
        m = cls(size_xyz=tuple(s * float(z["voxel"]) for s in shape), voxel=float(z["voxel"]),
                origin=tuple(float(v) for v in z["origin"]), l_lo=float(z["l_lo"]), l_hi=float(z["l_hi"]))
        m.logodds = z["logodds"].astype(np.float32)
        m.frames_integrated = int(z["frames"])
        m.eps_history = [float(v) for v in z["eps_history"]]
        return m
