"""Відновлення метричного масштабу монокулярної глибини.

Нейромережа повертає афінно-інваріантну обернену глибину r = a / Z + b
з невідомими a, b, які змінюються від кадру до кадру. За пікселями підлоги
та показами ToF-висотоміра відновлюються параметри лінійної моделі
1 / Z = alpha * r + beta (RANSAC, потім МНК на інлаєрах), обчислюється
самооцінка відносної похибки кадру eps, відкидаються "літаючі" пікселі
на розривах глибини.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    """Внутрішні параметри камери (модель pinhole) та її нахил."""
    width: int
    height: int
    hfov_deg: float = 70.0          # горизонтальне поле зору камери DJI Tello
    tilt_deg: float = 10.0          # сталий нахил оптичної осі вниз

    @property
    def fx(self):
        return self.width / (2.0 * np.tan(np.radians(self.hfov_deg) / 2.0))

    @property
    def K(self):
        f = self.fx
        return np.array([[f, 0.0, self.width / 2.0],
                         [0.0, f, self.height / 2.0],
                         [0.0, 0.0, 1.0]])

    def pixel_dirs(self):
        """Напрямки променів у системі камери (x вправо, y вниз, z вперед),
        нормовані так, що z = 1: точка = Z * d_c."""
        u, v = np.meshgrid(np.arange(self.width) + 0.5,
                           np.arange(self.height) + 0.5)
        f = self.fx
        dx = (u - self.width / 2.0) / f
        dy = (v - self.height / 2.0) / f
        return np.stack([dx, dy, np.ones_like(dx)], axis=-1)

    def rotation_world(self, yaw_deg):
        """Матриця повороту камера -> світ (x вперед, y вліво, z вгору)."""
        r0 = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])
        t = np.radians(self.tilt_deg)
        pitch = np.array([[np.cos(t), 0.0, np.sin(t)],
                          [0.0, 1.0, 0.0],
                          [-np.sin(t), 0.0, np.cos(t)]])
        y = np.radians(yaw_deg)
        yaw = np.array([[np.cos(y), -np.sin(y), 0.0],
                        [np.sin(y), np.cos(y), 0.0],
                        [0.0, 0.0, 1.0]])
        return yaw @ pitch @ r0


def floor_depth(cam, yaw_deg, height):
    """Глибина вздовж оптичної осі для пікселів, промінь яких перетинає
    площину підлоги z = 0 (камера на висоті height за висотоміром).
    Там, де промінь не перетинає підлогу, повертається нескінченність."""
    d_c = cam.pixel_dirs()
    R = cam.rotation_world(yaw_deg)
    d_w = d_c @ R.T
    dz = d_w[..., 2]
    Z = np.full(dz.shape, np.inf)
    down = dz < -1e-6
    Z[down] = -height / dz[down]
    return Z


def _fit_two(r1, z1, r2, z2):
    """Гіпотеза (alpha, beta) за двома пікселями підлоги."""
    inv1, inv2 = 1.0 / z1, 1.0 / z2
    if abs(r1 - r2) < 1e-9:
        return None
    alpha = (inv1 - inv2) / (r1 - r2)
    beta = inv1 - alpha * r1
    return alpha, beta


def recover_scale(rel_inv_depth, floor_mask, z_floor, tau=0.06, rho=0.2,
                  min_floor_frac=0.05, min_inlier_frac=0.01, n_iter=200, rng=None):
    """RANSAC-оцінка параметрів масштабу за пікселями підлоги.

    rel_inv_depth   -- обернена відносна глибина від нейромережі, H x W;
    floor_mask      -- булева маска опорної поверхні;
    z_floor         -- глибина пікселів підлоги за висотоміром (floor_depth);
    tau             -- поріг відносної похибки інлаєра;
    rho             -- мінімальна частка інлаєрів серед пікселів підлоги;
    min_floor_frac  -- мінімальна частка пікселів підлоги в кадрі;
    min_inlier_frac -- мінімальна частка інлаєрів від усіх пікселів кадру.
    Кадр відкидається, якщо підлоги майже не видно або інлаєрів замало:
    це захищає карту від кадрів з хибним масштабом.
    Повертає словник: accepted, alpha, beta, eps, n_floor, n_valid, n_inliers.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_pix = rel_inv_depth.size
    valid = floor_mask & np.isfinite(z_floor) & (z_floor > 0.2) & (z_floor < 12.0)
    r = rel_inv_depth[valid].astype(np.float64)
    z = z_floor[valid].astype(np.float64)
    n = r.size
    result = {"accepted": False, "alpha": None, "beta": None, "eps": None,
              "n_floor": int(floor_mask.sum()), "n_valid": int(n), "n_inliers": 0}
    min_pixels = max(50, int(min_inlier_frac * n_pix))
    if n < min_pixels or floor_mask.sum() < min_floor_frac * n_pix:
        return result
    best_inl = None
    best_cnt = 0
    for _ in range(n_iter):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        hyp = _fit_two(r[i], z[i], r[j], z[j])
        if hyp is None:
            continue
        alpha, beta = hyp
        inv = alpha * r + beta
        ok = inv > 1e-4
        err = np.full(n, np.inf)
        err[ok] = np.abs(1.0 / inv[ok] - z[ok]) / z[ok]
        inl = err < tau
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_inl is None or best_cnt < max(min_pixels, rho * n):
        result["n_inliers"] = best_cnt
        return result
    # уточнення методом найменших квадратів на інлаєрах: 1/z = alpha r + beta
    A = np.stack([r[best_inl], np.ones(best_cnt)], axis=1)
    sol, *_ = np.linalg.lstsq(A, 1.0 / z[best_inl], rcond=None)
    alpha, beta = float(sol[0]), float(sol[1])
    inv = alpha * r[best_inl] + beta
    good = inv > 1e-4
    rel_err = np.abs(1.0 / inv[good] - z[best_inl][good]) / z[best_inl][good]
    result.update({"accepted": True, "alpha": alpha, "beta": beta,
                   "eps": float(np.median(rel_err)), "n_inliers": best_cnt})
    return result


def metric_depth(rel_inv_depth, alpha, beta, z_min=0.2, z_max=12.0):
    """Метрична глибина кадру за параметрами лінійної моделі."""
    inv = alpha * rel_inv_depth + beta
    Z = np.full(inv.shape, np.nan)
    ok = inv > 1.0 / z_max
    Z[ok] = 1.0 / inv[ok]
    Z[(Z < z_min) | (Z > z_max)] = np.nan
    return Z


def flying_pixel_mask(Z, gamma=0.08):
    """Маска пікселів, які лишаються після відкидання "літаючих" точок:
    відносний градієнт глибини має бути меншим за gamma."""
    Zf = np.where(np.isfinite(Z), Z, np.nan)
    gx = np.abs(np.diff(Zf, axis=1, prepend=Zf[:, :1]))
    gy = np.abs(np.diff(Zf, axis=0, prepend=Zf[:1, :]))
    grad = np.fmax(gx, gy)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = grad / Zf
    keep = np.isfinite(Zf) & (rel < gamma)
    return keep


def backproject(cam, Z, yaw_deg, position, keep=None):
    """3D-точки у світовій системі для пікселів з відомою глибиною."""
    d_c = cam.pixel_dirs()
    R = cam.rotation_world(yaw_deg)
    pts = (d_c * Z[..., None]) @ R.T + np.asarray(position, dtype=np.float64)[None, None, :]
    mask = np.isfinite(Z) if keep is None else (keep & np.isfinite(Z))
    return pts[mask]


def process_frame(cam, rel_inv_depth, floor_mask, height, yaw_deg, position,
                  tau=0.06, rho=0.2, gamma=0.08, stride=2, rng=None):
    """Повна обробка одного кадру: масштаб, метрична глибина, фільтр
    "літаючих" пікселів, хмара точок у світовій системі."""
    z_floor = floor_depth(cam, yaw_deg, height)
    fit = recover_scale(rel_inv_depth, floor_mask, z_floor, tau=tau, rho=rho, rng=rng)
    if not fit["accepted"]:
        return fit, None, None
    Z = metric_depth(rel_inv_depth, fit["alpha"], fit["beta"])
    keep = flying_pixel_mask(Z, gamma=gamma)
    if stride > 1:
        sub = np.zeros_like(keep)
        sub[::stride, ::stride] = True
        keep_pts = keep & sub
    else:
        keep_pts = keep
    pts = backproject(cam, Z, yaw_deg, position, keep=keep_pts)
    return fit, Z, pts
