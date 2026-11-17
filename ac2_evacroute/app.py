"""Нейромережева оцінка прохідності завалів для людини та пошук безпечного
маршруту евакуації за 3D-картою зайнятості.

Команди:
  gendata  -- генерація синтетичних сцен, спостережених карт і міток
  train    -- навчання нейромережі прохідності TravNet
  evaluate -- порівняння правилової моделі та нейромережі на тестових сценах
  route    -- маршрут евакуації за картою першої програми (файл .npz)
  scene    -- рисунок для однієї тестової сцени
"""
import argparse
import json
import os
import time

import numpy as np

from map_layers import load_map_npz, column_layers, features_from_layers
from pose_model import (rule_state_map, truth_state_map, label_map, safety_margin,
                        WALK, CROUCH, CRAWL, BLOCKED, UNSEEN, STATE_NAMES)
from route_astar import plan_route, route_summary, check_route, classify_outcome, cells_near
import visualize as viz

VOXEL = 0.1
METHODS = {"М3": {"mu": 0.0, "u_max": 1.0, "kappa": 0.0},
           "М4": {"mu": None, "u_max": 0.9, "kappa": 6.0}}


# ---------------------------------------------------------------- gendata
def generate_split(seeds, rng, verbose=True, camera=None):
    """camera -- None (огляд по колу, як у синтетиці АС) або словник параметрів observe/hover_points,
    напр. scene_gen.RMTT_CAMERA (справжня камера RoboMaster TT)."""
    from scene_gen import make_scene, voxelize, hover_points, observe
    camera = dict(camera or {})
    height_range = camera.pop("height_range", (1.0, 1.6))
    X, Y, TS, TH, VIC, EXI = [], [], [], [], [], []
    t0 = time.perf_counter()
    for n, seed in enumerate(seeds):
        scene = make_scene(seed)
        occ = voxelize(scene)
        pts = hover_points(scene, rng, height_range=height_range)
        obs = observe(occ, pts, rng, **camera)
        layers = column_layers(obs, VOXEL)
        truth_state, truth_layers = truth_state_map(occ, VOXEL)              # з урахуванням ширини: для перевірки маршрутів
        truth_column, _ = truth_state_map(occ, VOXEL, check_width=False)     # стан колонки: мітки для нейромережі
        X.append(features_from_layers(layers).astype(np.float16))
        Y.append(label_map(truth_column, layers))
        TS.append(truth_state)
        TH.append(truth_layers["support"].astype(np.float16))
        VIC.append(scene["victim"])
        EXI.append(scene["exit"])
        if verbose and (n % 10 == 0 or n == len(seeds) - 1):
            print(f"  сцена {n + 1}/{len(seeds)} (seed {seed}): точок зависання {len(pts)}, "
                  f"оглянуто {100 * layers['observed'].mean():.0f} % вокселів, {time.perf_counter() - t0:.0f} с")
    return {"X": np.stack(X), "Y": np.stack(Y).astype(np.int8), "truth_state": np.stack(TS).astype(np.int8),
            "truth_support": np.stack(TH), "victim": np.asarray(VIC, dtype=np.float32),
            "exit": np.asarray(EXI, dtype=np.float32), "seeds": np.asarray(seeds)}


def cmd_gendata(a):
    os.makedirs(a.out, exist_ok=True)
    splits = {"train": range(0, a.n_train), "val": range(1000, 1000 + a.n_val),
              "test": range(2000, 2000 + a.n_test)}
    for name, seeds in splits.items():
        print(f"Вибірка {name}: {len(seeds)} сцен")
        rng = np.random.default_rng(a.seed + hash(name) % 1000)
        camera = None
        if a.camera == "rmtt":
            from scene_gen import RMTT_CAMERA
            camera = {**RMTT_CAMERA, "frame_bias": a.frame_bias, "rel_noise": a.rel_noise}
        data = generate_split(list(seeds), rng, camera=camera)
        np.savez_compressed(os.path.join(a.out, f"{name}.npz"), **data)
        print(f"  збережено {os.path.join(a.out, name + '.npz')}; розподіл міток:",
              {STATE_NAMES[k]: int(v) for k, v in enumerate(np.bincount(data['Y'].reshape(-1), minlength=5))})


# ---------------------------------------------------------------- train
def cmd_train(a):
    from nn_traversability import train
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    _, history = train(os.path.join(a.data, "train.npz"), os.path.join(a.data, "val.npz"), a.out,
                       epochs=a.epochs, batch=a.batch, lr=a.lr, width=a.width, device=a.device)
    if a.figs:
        os.makedirs(a.figs, exist_ok=True)
        viz.training_curves(history, out=os.path.join(a.figs, "training_curves.png"))


# ---------------------------------------------------------------- evaluate
def scene_routes(x, truth_state, truth_support, victim, exit_xy, model=None, eps=0.03, theta=0.6,
                 methods=METHODS, device="cpu"):
    """Маршрути методів М3, М4 і нейромережі для однієї сцени."""
    from nn_traversability import predict_probs, decide
    layers = {"support": x[0] * 3.0, "clearance": x[1] * 3.0, "unknown_walk": x[2], "unknown_crouch": x[3],
              "unknown_crawl": x[4], "density": x[5], "observed": x[6]}
    shape = truth_state.shape
    start = (int(victim[0] / VOXEL), int(victim[1] / VOXEL))
    exits = cells_near(exit_xy, VOXEL, 0.4, shape)
    truth_path = plan_route(truth_state, truth_support, np.zeros(shape), start, exits, VOXEL, kappa=0.0)
    truth_exists = truth_path is not None
    out = {"truth": {"path": truth_path, "outcome": "success" if truth_exists else "correct_refusal",
                     "summary": route_summary(truth_path, truth_state, truth_support, VOXEL) if truth_exists else None},
           "truth_exists": truth_exists, "states": {}}
    unknown = layers["unknown_crawl"]
    for key, name in (("m3", "М3"), ("m4", "М4")):
        p = methods[name]
        mu = safety_margin(eps) if p["mu"] is None else p["mu"]
        st = rule_state_map(layers, VOXEL, mu=mu, u_max=p["u_max"])
        path = plan_route(st, layers["support"], unknown, start, exits, VOXEL, kappa=p["kappa"])
        safe, t_true = check_route(path, truth_state, truth_support, VOXEL)
        out[key] = {"path": path, "outcome": classify_outcome(path, safe, truth_exists), "true_time": t_true,
                    "summary": route_summary(path, st, layers["support"], VOXEL) if path else None}
        out["states"][key] = st
    if model is not None:
        probs = predict_probs(model, x.astype(np.float32), device)
        st = decide(probs, theta=theta, unknown_crawl=unknown)
        path = plan_route(st, layers["support"], unknown, start, exits, VOXEL, kappa=methods["М4"]["kappa"])
        safe, t_true = check_route(path, truth_state, truth_support, VOXEL)
        out["nn"] = {"path": path, "outcome": classify_outcome(path, safe, truth_exists), "true_time": t_true,
                     "summary": route_summary(path, st, layers["support"], VOXEL) if path else None}
        out["states"]["nn"] = st
        out["probs"] = probs
    out["start"], out["exits"] = start, exits
    return out


def cmd_evaluate(a):
    from nn_traversability import load_model
    z = np.load(os.path.join(a.data, "test.npz"))
    model = load_model(a.model, a.device)
    n = z["X"].shape[0]
    counts = {m: {"success": 0, "dangerous": 0, "miss": 0, "correct_refusal": 0} for m in ("М3", "М4", "М5")}
    cell = {m: {"correct": 0, "total": 0, "false_passable": 0, "blocked": 0, "passable_found": 0, "passable_true": 0}
            for m in ("М3", "М4", "М5")}
    cm = np.zeros((5, 5), dtype=np.int64)
    n_exists, rel_time = 0, {m: [] for m in ("М3", "М4", "М5")}
    t0 = time.perf_counter()
    examples = []
    for i in range(n):
        x = z["X"][i].astype(np.float32)
        r = scene_routes(x, z["truth_state"][i], z["truth_support"][i].astype(np.float32), z["victim"][i],
                         z["exit"][i], model=model, eps=a.eps, theta=a.theta, device=a.device)
        n_exists += int(r["truth_exists"])
        truth = z["truth_state"][i]
        y = z["Y"][i]
        for key, m in (("m3", "М3"), ("m4", "М4"), ("nn", "М5")):
            counts[m][r[key]["outcome"]] += 1
            st = r["states"][key]
            seen = y != UNSEEN
            c = cell[m]
            c["correct"] += int(((st == truth) & seen).sum())
            c["total"] += int(seen.sum())
            c["false_passable"] += int(((st <= CRAWL) & (truth == BLOCKED)).sum())
            c["blocked"] += int((truth == BLOCKED).sum())
            c["passable_found"] += int(((st <= CRAWL) & (truth <= CRAWL)).sum())
            c["passable_true"] += int((truth <= CRAWL).sum())
            if r[key]["outcome"] == "success" and r["truth"]["summary"]:
                rel_time[m].append(r[key]["true_time"] / r["truth"]["summary"]["time_s"])
        pred = r["states"]["nn"]
        truth_cm = truth.copy()                      # точний стан з урахуванням ширини,
        truth_cm[y == UNSEEN] = UNSEEN               # невидимі дроном колонки -- «недостатньо оглянута»
        np.add.at(cm, (truth_cm.reshape(-1).astype(int), pred.reshape(-1).astype(int)), 1)
        if len(examples) < a.n_examples and r["truth_exists"] and r["nn"]["path"] is not None:
            examples.append((i, r))
        if i % 10 == 0:
            print(f"  сцена {i + 1}/{n}: М3 {r['m3']['outcome']}, М4 {r['m4']['outcome']}, М5 {r['nn']['outcome']}, "
                  f"{time.perf_counter() - t0:.0f} с")
    table = {}
    for m in counts:
        c = counts[m]
        table[m] = {"success": c["success"] / max(1, n_exists), "dangerous": c["dangerous"] / n,
                    "miss": c["miss"] / max(1, n_exists), "correct_refusal": c["correct_refusal"] / max(1, n - n_exists),
                    "rel_time": float(np.mean(rel_time[m])) if rel_time[m] else None,
                    "cell_accuracy": cell[m]["correct"] / max(1, cell[m]["total"]),
                    "false_passable_cells": cell[m]["false_passable"],
                    "false_passable_rate": cell[m]["false_passable"] / max(1, cell[m]["blocked"]),
                    "passable_recall": cell[m]["passable_found"] / max(1, cell[m]["passable_true"])}
    print(f"Тестових сцен {n}, маршрут існує у {n_exists} ({100 * n_exists / n:.1f} %)")
    print(f"{'Метод':6s} {'Успіх':>8s} {'Небезп.':>8s} {'Пропуск':>8s} {'Відмова':>8s} {'Відн.час':>9s} "
          f"{'Точн.кліт.':>10s} {'Хибно прох.':>12s} {'Повнота':>8s}")
    for m, t in table.items():
        rt = f"{t['rel_time']:.2f}" if t["rel_time"] else "-"
        print(f"{m:6s} {100 * t['success']:7.1f}% {100 * t['dangerous']:7.1f}% {100 * t['miss']:7.1f}% "
              f"{100 * t['correct_refusal']:7.1f}% {rt:>9s} {t['cell_accuracy']:10.3f} {t['false_passable_cells']:12d} "
              f"{t['passable_recall']:8.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"n_scenes": n, "n_exists": n_exists, "table": table, "confusion": cm.tolist(),
                   "counts": counts, "theta": a.theta, "eps": a.eps}, f, ensure_ascii=False, indent=1)
    if a.figs:
        os.makedirs(a.figs, exist_ok=True)
        viz.outcome_bars(table, out=os.path.join(a.figs, "outcomes.png"))
        viz.confusion_plot(cm, out=os.path.join(a.figs, "confusion.png"))
        for i, r in examples:
            viz.scene_panel(z["truth_state"][i], r["states"]["m3"], r["states"]["m4"], r["states"]["nn"],
                            {"truth": r["truth"], "m3": r["m3"], "m4": r["m4"], "nn": r["nn"]}, r["start"], r["exits"],
                            VOXEL, title=f"Тестова сцена {int(z['seeds'][i])}",
                            out=os.path.join(a.figs, f"scene_{int(z['seeds'][i])}.png"))


def cmd_scene(a):
    from nn_traversability import load_model
    z = np.load(os.path.join(a.data, "test.npz"))
    i = int(np.where(z["seeds"] == a.seed)[0][0])
    model = load_model(a.model, a.device) if a.model else None
    x = z["X"][i].astype(np.float32)
    r = scene_routes(x, z["truth_state"][i], z["truth_support"][i].astype(np.float32), z["victim"][i], z["exit"][i],
                     model=model, eps=a.eps, theta=a.theta, device=a.device)
    os.makedirs(a.figs, exist_ok=True)
    layers = {"support": x[0] * 3.0, "clearance": x[1] * 3.0, "unknown_crawl": x[4], "observed": x[6]}
    viz.layers_panel(layers, VOXEL, out=os.path.join(a.figs, f"layers_{a.seed}.png"))
    viz.scene_panel(z["truth_state"][i], r["states"]["m3"], r["states"]["m4"], r["states"].get("nn", r["states"]["m4"]),
                    {"truth": r["truth"], "m3": r["m3"], "m4": r["m4"], "nn": r.get("nn")}, r["start"], r["exits"],
                    VOXEL, title=f"Тестова сцена {a.seed}", out=os.path.join(a.figs, f"scene_{a.seed}.png"))
    for key in ("truth", "m3", "m4", "nn"):
        if key in r:
            print(key, r[key]["outcome"], r[key]["summary"])


# ---------------------------------------------------------------- route (карта першої програми)
def cmd_route(a):
    from nn_traversability import load_model, predict_probs, decide
    state3d, voxel, eps = load_map_npz(a.map)
    print(f"Карта {state3d.shape}, воксел {voxel} м, медіана eps = {eps:.3f}")
    layers = column_layers(state3d, voxel)
    x = features_from_layers(layers)
    model = load_model(a.model, a.device)
    t0 = time.perf_counter()
    probs = predict_probs(model, x, a.device)
    st_nn = decide(probs, theta=a.theta, unknown_crawl=layers["unknown_crawl"])
    t_nn = time.perf_counter() - t0
    mu = safety_margin(eps if eps > 0 else a.eps)
    st_m3 = rule_state_map(layers, voxel, mu=0.0, u_max=1.0)
    st_m4 = rule_state_map(layers, voxel, mu=mu, u_max=0.9)
    start = (int(a.start[0] / voxel), int(a.start[1] / voxel))
    exits = cells_near(a.exit, voxel, 0.4, st_nn.shape)
    results = {}
    for key, st, kappa in (("m3", st_m3, 0.0), ("m4", st_m4, 6.0), ("nn", st_nn, 6.0)):
        t0 = time.perf_counter()
        path = plan_route(st, layers["support"], layers["unknown_crawl"], start, exits, voxel, kappa=kappa)
        dt = time.perf_counter() - t0
        summ = route_summary(path, st, layers["support"], voxel) if path else None
        results[key] = {"path": path, "outcome": "success" if path else "miss", "summary": summ}
        print(f"{key}: {'маршрут знайдено' if path else 'безпечного маршруту немає'}; A* {1000 * dt:.0f} мс; {summ}")
    print(f"Нейромережа: {1000 * t_nn:.0f} мс; розподіл станів:",
          {STATE_NAMES[k]: int(v) for k, v in enumerate(np.bincount(st_nn.reshape(-1), minlength=5))})
    os.makedirs(a.figs, exist_ok=True)
    viz.layers_panel(layers, voxel, out=os.path.join(a.figs, "map_layers.png"))
    viz.methods_panel({"m3": st_m3, "m4": st_m4, "nn": st_nn}, results, start, exits, voxel,
                      title=a.title or os.path.basename(a.map), out=os.path.join(a.figs, "route_panel.png"))
    with open(os.path.join(a.figs, "route_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: {"outcome": v["outcome"], "summary": v["summary"],
                       "path": [list(map(int, c)) for c in v["path"]] if v["path"] else None}
                   for k, v in results.items()}, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("gendata")
    s.add_argument("--out", required=True)
    s.add_argument("--n-train", type=int, default=160)
    s.add_argument("--n-val", type=int, default=30)
    s.add_argument("--n-test", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--camera", choices=("sim", "rmtt"), default="sim",
                   help="sim -- огляд по колу (як у АС); rmtt -- справжня камера RoboMaster TT")
    s.add_argument("--frame-bias", type=float, default=0.10, help="для rmtt: розкид масштабу кадру")
    s.add_argument("--rel-noise", type=float, default=0.08, help="для rmtt: шум дальності променя")
    s.set_defaults(func=cmd_gendata)

    s = sub.add_parser("train")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--width", type=int, default=16)
    s.add_argument("--device", default="cpu")
    s.add_argument("--figs")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate")
    s.add_argument("--data", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--figs")
    s.add_argument("--eps", type=float, default=0.03)
    s.add_argument("--theta", type=float, default=0.6)
    s.add_argument("--n-examples", type=int, default=3)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("scene")
    s.add_argument("--data", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--model")
    s.add_argument("--figs", required=True)
    s.add_argument("--eps", type=float, default=0.03)
    s.add_argument("--theta", type=float, default=0.6)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_scene)

    s = sub.add_parser("route")
    s.add_argument("--map", required=True, help="карта .npz першої програми")
    s.add_argument("--model", required=True)
    s.add_argument("--start", type=float, nargs=2, required=True, help="x y постраждалого, м")
    s.add_argument("--exit", type=float, nargs=2, required=True, help="x y виходу, м")
    s.add_argument("--figs", required=True)
    s.add_argument("--eps", type=float, default=0.03)
    s.add_argument("--theta", type=float, default=0.6)
    s.add_argument("--title")
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_route)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
