"""Візуалізація карт прохідності, маршрутів, кривих навчання та матриці помилок."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pose_model import STATE_NAMES, WALK, CROUCH, CRAWL, BLOCKED, UNSEEN

STATE_COLORS = ["#2ca02c", "#ffcc00", "#ff7f0e", "#202020", "#bdbdbd"]
STATE_CMAP = ListedColormap(STATE_COLORS)
ROUTE_COLORS = {WALK: "#00ff2a", CROUCH: "#ffff00", CRAWL: "#ff2a00"}


def _extent(shape, voxel):
    return [0, shape[0] * voxel, 0, shape[1] * voxel]


def draw_state(ax, state, voxel=0.1, title=None, legend=False):
    ax.imshow(state.T, origin="lower", cmap=STATE_CMAP, vmin=-0.5, vmax=4.5, extent=_extent(state.shape, voxel),
              interpolation="nearest")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(STATE_COLORS, STATE_NAMES)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)


def draw_route(ax, path, state, voxel=0.1, start=None, exits=None, lw=2.5):
    if path:
        for a, b in zip(path[:-1], path[1:]):
            k = int(state[b])
            ax.plot([(a[0] + 0.5) * voxel, (b[0] + 0.5) * voxel], [(a[1] + 0.5) * voxel, (b[1] + 0.5) * voxel],
                    color=ROUTE_COLORS.get(k, "#ff00ff"), lw=lw, solid_capstyle="round")
    if start is not None:
        ax.plot((start[0] + 0.5) * voxel, (start[1] + 0.5) * voxel, "o", ms=9, mfc="#1f77b4", mec="white", label="постраждалий")
    if exits:
        ex = np.asarray(exits)
        ax.plot((ex[:, 0].mean() + 0.5) * voxel, (ex[:, 1].mean() + 0.5) * voxel, "s", ms=9, mfc="#d62728", mec="white", label="вихід")


def scene_panel(truth_state, obs_state, rule_state, nn_state, routes, start, exits, voxel=0.1,
                title=None, out=None):
    """Чотири карти: точна, спостережена (правила без невизначеності),
    правила з невизначеністю, нейромережа; маршрути накладено."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    panels = [("Точна модель сцени", truth_state), ("Правила без невизначеності (М3)", obs_state),
              ("Правила з запасами та штрафом (М4)", rule_state), ("Нейромережа прохідності (М5)", nn_state)]
    for ax, (name, st), key in zip(axes, panels, ("truth", "m3", "m4", "nn")):
        draw_state(ax, st, voxel, title=name)
        r = routes.get(key)
        draw_route(ax, r["path"] if r else None, st, voxel, start=start, exits=exits)
        if r is not None:
            lab = {"success": "безпечний маршрут", "dangerous": "НЕБЕЗПЕЧНИЙ маршрут", "miss": "маршрут не знайдено",
                   "correct_refusal": "коректна відмова"}[r["outcome"]]
            txt = lab if r["path"] is None else f"{lab}, {r['summary']['time_s']:.0f} с"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    axes[0].legend(handles=[Patch(color=c, label=n) for c, n in zip(STATE_COLORS, STATE_NAMES)] +
                   [Patch(color=ROUTE_COLORS[WALK], label="маршрут: пішки"), Patch(color=ROUTE_COLORS[CROUCH], label="маршрут: пригнувшись"),
                    Patch(color=ROUTE_COLORS[CRAWL], label="маршрут: повзком")],
                   loc="upper center", bbox_to_anchor=(2.3, -0.1), ncol=4, fontsize=9, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def methods_panel(states, routes, start, exits, voxel=0.1, title=None, out=None):
    """Карти станів методів для реальної карти (без точної моделі): по панелі на метод."""
    names = {"m3": "Правила без невизначеності (М3)", "m4": "Правила з запасами та штрафом (М4)",
             "nn": "Нейромережа прохідності (М5)"}
    keys = [k for k in ("m3", "m4", "nn") if k in states]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.9 * len(keys), 5.4))
    for ax, key in zip(np.atleast_1d(axes), keys):
        st = states[key]
        draw_state(ax, st, voxel, title=names[key])
        r = routes.get(key)
        draw_route(ax, r["path"] if r else None, st, voxel, start=start, exits=exits)
        if r is not None:
            txt = "безпечного маршруту немає" if r["path"] is None else \
                f"маршрут знайдено, {r['summary']['time_s']:.0f} с"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(STATE_COLORS, STATE_NAMES)] +
               [Patch(color=ROUTE_COLORS[WALK], label="маршрут: пішки"),
                Patch(color=ROUTE_COLORS[CROUCH], label="маршрут: пригнувшись"),
                Patch(color=ROUTE_COLORS[CRAWL], label="маршрут: повзком")],
               loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def layers_panel(layers, voxel=0.1, out=None):
    """Шари спостереженої карти: опора, просвіт, невідоме у смузі повзком, оглянутість."""
    items = [("Висота опори, м", layers["support"], "viridis", 0, 1.5),
             ("Вільний просвіт, м", layers["clearance"], "viridis", 0, 3.0),
             ("Частка невідомих (смуга «повзком»)", layers["unknown_crawl"], "magma", 0, 1),
             ("Частка оглянутих вокселів", layers["observed"], "magma", 0, 1)]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, (name, arr, cmap, vmin, vmax) in zip(axes, items):
        im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=_extent(arr.shape, voxel))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x, м")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("y, м")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def training_curves(history, out=None):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="навчальна вибірка")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="валідаційна вибірка")
    ax[0].set_xlabel("епоха")
    ax[0].set_ylabel("функція втрат")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [100 * h["val_acc"] for h in history], "o-", label="точність за клітинками, %")
    ax[1].plot(ep, [100 * h["val_false_passable"] for h in history], "s-", color="#d62728",
               label="хибно прохідні серед непрохідних, %")
    ax[1].set_xlabel("епоха")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def confusion_plot(cm, out=None, title="Матриця помилок нейромережі (тестові сцени)"):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.1f} %", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STATE_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(STATE_NAMES, fontsize=8)
    ax.set_xlabel("передбачений стан")
    ax.set_ylabel("точний стан")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def outcome_bars(table, out=None):
    """Стовпчики результатів побудови маршрутів для методів."""
    methods = list(table.keys())
    keys = [("success", "успіх", "#2ca02c"), ("dangerous", "небезпечний маршрут", "#d62728"),
            ("miss", "пропуск", "#ff7f0e"), ("correct_refusal", "коректна відмова", "#1f77b4")]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 4))
    wdt = 0.2
    for k, (key, lab, col) in enumerate(keys):
        vals = [100 * table[m][key] for m in methods]
        ax.bar(x + (k - 1.5) * wdt, vals, wdt, label=lab, color=col)
        for xi, v in zip(x + (k - 1.5) * wdt, vals):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("% сцен")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
