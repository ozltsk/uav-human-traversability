"""Візуалізація результатів: панель кадру, вигляд карти згори, зрізи та
3D-подання зайнятих вокселів."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from occupancy_map import FREE, UNKNOWN, OCCUPIED

STATE_CMAP = ListedColormap(["#ffffff", "#b8b8b8", "#202020"])   # вільно, невідомо, зайнято


def frame_panel(image, rel_inv_depth, floor_mask, Z=None, fit=None, keep=None,
                title=None, out=None):
    """Чотири панелі: кадр, обернена глибина, маска підлоги, метрична глибина."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    ax[0].imshow(image)
    ax[0].set_title("Кадр камери")
    im = ax[1].imshow(rel_inv_depth, cmap="magma")
    ax[1].set_title("Обернена відносна глибина")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    over = np.asarray(image).copy()
    over[floor_mask] = (0.45 * over[floor_mask] + 0.55 * np.array([40, 200, 60])).astype(np.uint8)
    ax[2].imshow(over)
    ax[2].set_title(f"Опорна поверхня: {100.0 * floor_mask.mean():.1f} % пікселів")
    if Z is not None and fit is not None and fit.get("accepted"):
        Zs = np.where(keep, Z, np.nan) if keep is not None else Z
        im = ax[3].imshow(Zs, cmap="viridis", vmin=0, vmax=np.nanpercentile(Zs, 98))
        plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.02, label="м")
        ax[3].set_title(f"Метрична глибина, eps = {fit['eps']:.3f}, інлаєрів {fit['n_inliers']}")
    else:
        ax[3].imshow(np.zeros_like(rel_inv_depth), cmap="gray", vmin=0, vmax=1)
        n_fl = fit.get("n_floor", 0) if fit else 0
        n_in = fit.get("n_inliers", 0) if fit else 0
        ax[3].text(0.5, 0.5, f"Кадр відкинуто:\nпідлога {n_fl} пікс.,\nінлаєрів {n_in}",
                   ha="center", va="center", transform=ax[3].transAxes, color="white", fontsize=12)
        ax[3].set_title("Масштаб не відновлено")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def map_top_view(occ_map, log=None, z_from=0.3, z_to=1.8, truth=None, out=None, title=None):
    """Вигляд карти згори (смуга висот z_from..z_to) з точками зависання."""
    top = occ_map.top_view(z_from, z_to)
    ncol = 2 if truth is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 6))
    axes = np.atleast_1d(axes)
    ext = [occ_map.origin[0], occ_map.origin[0] + occ_map.shape[0] * occ_map.voxel,
           occ_map.origin[1], occ_map.origin[1] + occ_map.shape[1] * occ_map.voxel]
    axes[0].imshow(top.T, origin="lower", cmap=STATE_CMAP, vmin=0, vmax=2, extent=ext)
    axes[0].set_title(f"Карта зайнятості, смуга {z_from}-{z_to} м\n(біле - вільно, сіре - невідомо, чорне - зайнято)")
    if log is not None:
        wp = {}
        for r in log.records:
            wp.setdefault(r.waypoint, (r.x, r.y))
            dx, dy = np.cos(np.radians(r.yaw_deg)), np.sin(np.radians(r.yaw_deg))
            axes[0].arrow(r.x, r.y, 0.35 * dx, 0.35 * dy, head_width=0.12, color="#d62728", lw=0.8)
        xs, ys = zip(*wp.values())
        axes[0].plot(xs, ys, "o-", color="#d62728", ms=4, lw=0.8, label="точки зависання")
        axes[0].legend(loc="upper right", fontsize=8)
    if truth is not None:
        k0, k1 = int(z_from / occ_map.voxel), int(z_to / occ_map.voxel)
        t = truth[:, :, k0:k1].any(axis=2)
        axes[1].imshow(t.T, origin="lower", cmap="gray_r", extent=ext)
        axes[1].set_title("Точна модель сцени (та сама смуга висот)")
    for a in axes:
        a.set_xlabel("x, м")
        a.set_ylabel("y, м")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def map_slices(occ_map, heights=(0.3, 0.9, 1.5, 2.1), out=None):
    """Горизонтальні зрізи карти на заданих висотах."""
    s = occ_map.state()
    fig, axes = plt.subplots(1, len(heights), figsize=(4.2 * len(heights), 4.2))
    ext = [occ_map.origin[0], occ_map.origin[0] + occ_map.shape[0] * occ_map.voxel,
           occ_map.origin[1], occ_map.origin[1] + occ_map.shape[1] * occ_map.voxel]
    for a, h in zip(np.atleast_1d(axes), heights):
        k = min(int(h / occ_map.voxel), occ_map.shape[2] - 1)
        a.imshow(s[:, :, k].T, origin="lower", cmap=STATE_CMAP, vmin=0, vmax=2, extent=ext)
        a.set_title(f"z = {h:.1f} м")
        a.set_xlabel("x, м")
    np.atleast_1d(axes)[0].set_ylabel("y, м")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def map_3d(occ_map, out=None, max_points=60000, elev=38, azim=-125):
    """Зайняті воксели у тривимірному вигляді, колір за висотою."""
    s = occ_map.state()
    idx = np.argwhere(s == OCCUPIED)
    if idx.shape[0] > max_points:
        idx = idx[np.random.default_rng(0).choice(idx.shape[0], max_points, replace=False)]
    pts = occ_map.cell_center(idx)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=2, marker="s")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_zlabel("z, м")
    ax.set_box_aspect((occ_map.shape[0], occ_map.shape[1], occ_map.shape[2] * 1.5))
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.08, label="висота, м")
    ax.set_title(f"Зайняті воксели: {int((s == OCCUPIED).sum())}, кадрів {occ_map.frames_integrated}")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def frame_table_plot(rows, out=None):
    """Самооцінка похибки eps і кількість інлаєрів для кадрів обльоту."""
    idx = [r["index"] for r in rows]
    eps = [r["eps"] if r["accepted"] else np.nan for r in rows]
    inl = [r["n_inliers"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(10, 3.4))
    ax1.bar(idx, inl, color="#9ecae1", label="інлаєри підлоги")
    ax1.set_xlabel("номер кадру")
    ax1.set_ylabel("інлаєрів, пікс.")
    ax2 = ax1.twinx()
    ax2.plot(idx, eps, "o-", color="#d62728", ms=3, lw=1, label="самооцінка eps")
    ax2.set_ylabel("eps")
    rej = [r["index"] for r in rows if not r["accepted"]]
    for x in rej:
        ax1.axvline(x, color="k", ls=":", lw=0.8)
    ax1.set_title(f"Кадрів {len(rows)}, відкинуто {len(rej)}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
