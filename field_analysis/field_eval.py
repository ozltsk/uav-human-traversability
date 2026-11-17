"""Натурна перевірка «Рятівного маршруту»: карта АС Р1 і прохідність АС Р2 на справжніх обльотах
проти виміряних проходів (ground_truth виконавиці, 03-04.11.2026).

  python field_eval.py --logs ../runs/real_20261104 --gt ../data/field_20261104/ground_truth \
                       --out ../runs/real_20261104/eval [--skip-build]

Для кожного журналу зі списку _selected.txt:
  1) АС Р1 buildmap -> карта .npz і рисунки АС Р1;
  2) шари колонок і стани прохідності: М3 (правила), М4 (правила із запасом), М5 (TravNet, theta 0,6);
  3) кожен виміряний прохід: очікуваний стан за правилами пози АС Р2 з виміряних ширини й просвіту
     проти розподілу станів у смузі ±0,3 м навколо отвору;
  4) маршрут A* від точки за бар'єром до скляної стіни з дверима;
  5) рисунок: карта зайнятості й стани М5 з накладеними предметами, проходами й точками облету.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ac2_evacroute"))
sys.path.insert(0, HERE)
from map_layers import load_map_npz, column_layers, features_from_layers   # noqa: E402
from pose_model import rule_state_map, safety_margin, POSES, WALK, CROUCH, CRAWL, BLOCKED, UNSEEN  # noqa: E402
from route_astar import plan_route, route_summary, cells_near                # noqa: E402

STATE_UA = ("пішки", "пригнувшись", "повзком", "непрохідний", "не оглянуто")
START, EXIT = (3.8, 4.0), (0.6, 2.4)     # постраждалий за бар'єром (бік вікон); вихід -- скляна стіна з дверима


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delim = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    rows = []
    for r in csv.DictReader(text.splitlines(), delimiter=delim):
        rows.append({k: (v or "").strip() for k, v in r.items() if k})
    return rows


def num(v):
    try:
        return float(v.replace(",", "."))
    except (AttributeError, ValueError):
        return None


def expected_state(width, clearance):
    """Найшвидша поза АС Р2 (висота H, ширина W), для якої проходять виміряні ширина й просвіт."""
    for k in (WALK, CROUCH, CRAWL):
        _name, H, W, _v = POSES[k]
        if (clearance is None or clearance >= H) and (width is None or width >= W):
            return k
    return BLOCKED


def band_cells(p, voxel, shape, half=0.3, trim=0.15):
    x0, y0, x1, y1 = p
    seg = np.array([x1 - x0, y1 - y0])
    L = float(np.hypot(*seg))
    u = seg / L
    n = np.array([-u[1], u[0]])
    cells = set()
    for t in np.arange(trim * L, (1 - trim) * L + 1e-9, 0.05):
        for s in np.arange(-half, half + 1e-9, 0.05):
            x, y = np.array([x0, y0]) + t * u + s * n
            i, j = int(x / voxel), int(y / voxel)
            if 0 <= i < shape[0] and 0 <= j < shape[1]:
                cells.add((i, j))
    return sorted(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "runs", "ac2_models", "travnet.pt"))
    ap.add_argument("--model-rmtt", default=os.path.join(ROOT, "runs", "ac2_models_rmtt", "travnet.pt"),
                    help="TravNet, навчена на синтетиці з камерою RMTT (М5р); порожньо -- без неї")
    ap.add_argument("--list", default="_selected.txt", help="файл зі списком журналів у теці --logs")
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--max-eps", type=float, default=None, help="поріг похибки підлоги для АС Р1 buildmap")
    ap.add_argument("--build", choices=("ac1", "contact", "metric", "metric_contact"), default="ac1",
                    help="ac1 -- buildmap АС Р1; contact -- + шар перешкод за контактом; metric -- метрична глибина")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from nn_traversability import load_model, predict_probs, decide
    model = load_model(a.model, "cpu")
    model_r = load_model(a.model_rmtt, "cpu") if a.model_rmtt and os.path.exists(a.model_rmtt) else None
    with open(os.path.join(a.logs, a.list), encoding="utf-8") as f:
        logs = [s.strip() for s in f if s.strip()]
    report, passage_rows = [], []
    for name in logs:
        scene = name.split("_")[0]
        log_dir = os.path.join(a.logs, name)
        npz = os.path.join(a.out, f"{name}.npz")
        if (not a.skip_build or not os.path.exists(npz)) and a.build != "ac1":
            from contact_layer import build_map_plus
            build_map_plus(log_dir, npz, max_eps=a.max_eps, contact="contact" in a.build,
                           depth="metric" if a.build.startswith("metric") else "rel")
        elif not a.skip_build or not os.path.exists(npz):
            cmd = [sys.executable, os.path.join(ROOT, "ac1_evacmap", "app.py"), "buildmap",
                   "--log", log_dir, "--out", npz, "--figs", os.path.join(a.out, f"ac1_{name}")]
            if a.max_eps is not None:
                cmd += ["--max-eps", str(a.max_eps)]
            subprocess.run(cmd,
                           check=True, stdout=subprocess.DEVNULL, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        with open(os.path.splitext(npz)[0] + "_summary.json", encoding="utf-8") as f:
            rep_ac1 = json.load(f)
        state3d, voxel, eps = load_map_npz(npz)
        layers = column_layers(state3d, voxel)
        feats = features_from_layers(layers)
        probs = predict_probs(model, feats, "cpu")
        st = {"М5": decide(probs, theta=a.theta, unknown_crawl=layers["unknown_crawl"]),
              "М3": rule_state_map(layers, voxel, mu=0.0, u_max=1.0),
              "М4": rule_state_map(layers, voxel, mu=safety_margin(eps if eps > 0 else 0.03), u_max=0.9)}
        if model_r is not None:
            st["М5р"] = decide(predict_probs(model_r, feats, "cpu"), theta=a.theta, unknown_crawl=layers["unknown_crawl"])
            # М6: клітинка дозволена лише тоді, коли її пропускають і правила із запасом, і мережа;
            # стан -- повільніша з двох поз (більший код), невизначеність -- з М4
            m4, m5 = st["М4"], st["М5р"]
            comb = np.maximum(m4, m5).astype(np.int8)
            comb[(m4 == UNSEEN) | (m5 == UNSEEN)] = UNSEEN
            comb[(m4 == BLOCKED) | (m5 == BLOCKED)] = BLOCKED
            st["М6"] = comb
        objs = [o for o in read_csv(os.path.join(a.gt, f"{scene}_objects.csv")) if num(o.get("x0")) is not None]
        pas = [p for p in read_csv(os.path.join(a.gt, f"{scene}_passages.csv")) if num(p.get("y0")) is not None]
        shape = st["М5"].shape
        for p in pas:
            seg = (num(p["x0"]), num(p["y0"]), num(p["x1"]), num(p["y1"]))
            exp = expected_state(num(p.get("width_m")), num(p.get("clearance_m")))
            cells = band_cells(seg, voxel, shape)
            row = {"scene": scene, "log": name, "passage": p["passage_id"], "width_m": p.get("width_m"),
                   "clearance_m": p.get("clearance_m"), "expected": STATE_UA[exp], "cells": len(cells)}
            for m, s in st.items():
                vals = np.array([s[c] for c in cells])
                frac = np.bincount(vals, minlength=5) / max(len(vals), 1)
                seen = vals[vals != UNSEEN]
                best = int(np.bincount(seen, minlength=5).argmax()) if len(seen) else UNSEEN
                row[f"{m}_мода"] = STATE_UA[best]
                row[f"{m}_розподіл"] = " ".join(f"{STATE_UA[k][:4]} {100 * frac[k]:.0f}%" for k in range(5) if frac[k] > 0)
            passage_rows.append(row)
        start = (int(START[0] / voxel), int(START[1] / voxel))
        exits = cells_near(EXIT, voxel, 0.4, shape)
        routes = {}
        for m, s in st.items():
            kappa = 0.0 if m == "М3" else 6.0
            path = plan_route(s, layers["support"], layers["unknown_crawl"], start, exits, voxel, kappa=kappa)
            routes[m] = path
        counts = {m: {STATE_UA[k]: int(v) for k, v in enumerate(np.bincount(s.reshape(-1), minlength=5))}
                  for m, s in st.items()}
        report.append({"log": name, "scene": scene, "frames_accepted": rep_ac1.get("frames_accepted"),
                       "frames_total": rep_ac1.get("frames_total"), "eps_median": rep_ac1.get("eps_median"),
                       "occupied": rep_ac1.get("occupied"), "free": rep_ac1.get("free"),
                       "unknown": rep_ac1.get("unknown"), "states": counts,
                       "routes": {m: (route_summary(p, st[m], layers["support"], voxel) if p else None)
                                  for m, p in routes.items()},
                       "route_cross_y": {m: _cross_y(p, voxel) for m, p in routes.items()},
                       "paths": {m: ([[int(i), int(j)] for i, j in p] if p else None) for m, p in routes.items()},
                       "voxel": voxel, "map_quality": map_quality(state3d, voxel, objs)})
        best = "М5р" if "М5р" in st else "М5"
        figure(name, scene, state3d, st[best], voxel, objs, pas, log_dir, routes[best],
               os.path.join(a.out, f"{name}.png"), method=best)
        print(f"{name}: кадрів прийнято {rep_ac1.get('frames_accepted')} з {rep_ac1.get('frames_total')}, "
              + ", ".join(f"маршрут {m} {'є' if routes[m] else 'немає'}" for m in st))
    with open(os.path.join(a.out, "passages.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(passage_rows[0].keys()))
        w.writeheader()
        w.writerows(passage_rows)
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    for r in passage_rows:
        print(f"{r['scene']} {r['log'][3:5]} {r['passage']:7} очікувано {r['expected']:12} "
              + "; ".join(f"{m}: {r[m + '_мода']}" for m in ("М3", "М4", "М5", "М5р") if m + "_мода" in r))


ZONE = (0.8, 0.9, 4.4, 5.0)       # центральна зона кімнати: меблі вздовж стін не виміряні


def map_quality(state3d, voxel, objs, zone=ZONE, tol_m=0.25):
    """Якість карти проти виміряних предметів: повнота (частка клітинок основ предметів, біля яких
    у межах tol_m є зайнята клітинка карти на 0,1-1,9 м) і точність (частка зайнятих клітинок зони,
    що лежать у межах tol_m від якогось предмета). Предмети з просвітом (стіл, жердина) теж предмети."""
    from scipy.ndimage import distance_transform_edt
    from map_layers import OCCUPIED
    nz = state3d.shape[2]
    occ = (state3d[:, :, int(0.1 / voxel):min(nz, int(1.9 / voxel))] == OCCUPIED).any(axis=2)
    gt = np.zeros(occ.shape, bool)
    for o in objs:
        x0, y0, x1, y1 = (num(o[k]) for k in ("x0", "y0", "x1", "y1"))
        if None in (x0, y0, x1, y1):
            continue
        i0, i1 = int(min(x0, x1) / voxel), int(np.ceil(max(x0, x1) / voxel))
        j0, j1 = int(min(y0, y1) / voxel), int(np.ceil(max(y0, y1) / voxel))
        gt[max(0, i0):i1, max(0, j0):j1] = True
    if not gt.any():
        return None
    zx0, zy0, zx1, zy1 = (int(v / voxel) for v in zone)
    zmask = np.zeros(occ.shape, bool)
    zmask[zx0:zx1, zy0:zy1] = True
    d_to_occ = distance_transform_edt(~occ) * voxel
    d_to_gt = distance_transform_edt(~gt) * voxel
    recall = float((d_to_occ[gt] <= tol_m).mean())
    occ_z = occ & zmask
    precision = float((d_to_gt[occ_z] <= tol_m).mean()) if occ_z.any() else None
    return {"object_recall": round(recall, 3), "occupied_precision": None if precision is None else round(precision, 3),
            "occupied_cells_zone": int(occ_z.sum()), "object_cells": int(gt.sum())}


def _cross_y(path, voxel, x_bar=2.45):
    """y, на якому маршрут перетинає лінію бар'єра x = 2,45 м (None -- маршруту немає)."""
    if not path:
        return None
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        xa, xb = (i0 + 0.5) * voxel, (i1 + 0.5) * voxel
        if (xa - x_bar) * (xb - x_bar) <= 0:
            return round((j0 + 0.5) * voxel, 2)
    return None


def figure(name, scene, state3d, st, voxel, objs, pas, log_dir, route, out, method="М5"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from map_layers import OCCUPIED, FREE
    nz = state3d.shape[2]
    band = state3d[:, :, int(0.1 / voxel):min(nz, int(1.9 / voxel))]
    occ = (band == OCCUPIED).any(axis=2)
    free = (band == FREE).any(axis=2)
    top = np.where(occ, 2, np.where(free, 0, 1))
    tel = list(csv.DictReader(open(os.path.join(log_dir, "telemetry.csv"), encoding="utf-8")))
    wp = {}
    for r in tel:
        wp[int(r["waypoint"])] = (float(r["x"]), float(r["y"]))
    ext = [0, st.shape[0] * voxel, 0, st.shape[1] * voxel]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    panels = ((top, ListedColormap(["#ffffff", "#c8c8c8", "#333333"]), "АС Р1: зайнято / вільно / невідомо (0,1-1,9 м)"),
              (st, ListedColormap(["#4caf50", "#ffd54f", "#ff8a3d", "#8b1a1a", "#bdbdbd"]),
               f"АС Р2 ({method}): пішки / пригнувшись / повзком / непрохідна / не оглянуто"))
    for ax, (img, cmap, title) in zip(axes, panels):
        ax.imshow(img.T, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=cmap.N - 1, interpolation="nearest")
        for o in objs:
            x0, y0, x1, y1 = (num(o[k]) for k in ("x0", "y0", "x1", "y1"))
            if None in (x0, y0, x1, y1):
                continue
            ax.add_patch(Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0), fill=False,
                                   edgecolor="#d81b60", lw=1.6))
        for p in pas:
            xs = [num(p["x0"]), num(p["x1"])]
            ys = [num(p["y0"]), num(p["y1"])]
            ax.plot(xs, ys, color="#1565c0", lw=3, alpha=0.8)
            ax.text(xs[1] + 0.05, ys[1], p["passage_id"], color="#1565c0", fontsize=8)
        for k, (x, y) in sorted(wp.items()):
            ax.plot(x, y, "o", ms=7, mfc="#fdd835", mec="black")
            ax.text(x + 0.08, y + 0.08, str(k + 1), fontsize=8)
        if route:
            ax.plot([(i + 0.5) * voxel for i, _ in route], [(j + 0.5) * voxel for _, j in route], color="#00acc1", lw=2)
        ax.plot(*START, "*", ms=12, color="#e53935")
        ax.plot(*EXIT, "s", ms=8, color="#1e88e5")
        ax.set_xlabel("x, м (від скляної стіни)")
        ax.set_ylabel("y, м (від стіни з вішаками)")
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{name}: рожеве -- виміряні предмети, синє -- проходи, жовте -- точки облету (з рискання)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
