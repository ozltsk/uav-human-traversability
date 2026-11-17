"""Пошук безпечного маршруту евакуації алгоритмом A*.

Сітка клітинок розглядається як граф з 8-зв'язністю. Вартість переходу --
час руху з урахуванням пози в клітинці призначення, підйому на уламки та
штрафу за недосліджені ділянки: t = (d / v_k + max(0, dh) / v_climb) *
(1 + kappa * u), де u -- частка невідомих вокселів у смузі пози. Перехід
заборонений, якщо уступ вищий за MAX_STEP. Евристика -- евклідова відстань
до множини виходу, поділена на найбільшу швидкість; її для всіх клітинок
одночасно обчислює перетворення відстані, тому вона допустима й монотонна.
"""
import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt

from pose_model import SPEEDS, V_CLIMB, MAX_STEP, WALK, CROUCH, CRAWL, BLOCKED, UNSEEN

NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))]


def cells_near(xy, voxel, radius, shape):
    """Індекси клітинок у квадраті radius навколо точки xy."""
    i0, j0 = int(xy[0] / voxel), int(xy[1] / voxel)
    r = int(np.ceil(radius / voxel))
    cells = [(i, j) for i in range(i0 - r, i0 + r + 1) for j in range(j0 - r, j0 + r + 1)
             if 0 <= i < shape[0] and 0 <= j < shape[1]]
    return cells


def plan_route(state, support, unknown, start, exits, voxel=0.1, kappa=6.0):
    """A* від клітинки start до будь-якої клітинки з exits.
    state -- карта станів (0..4), support -- висота опори, unknown -- частка
    невідомих вокселів (для штрафу). Повертає список клітинок маршруту або None."""
    nx, ny = state.shape
    passable = state <= CRAWL
    exit_mask = np.zeros_like(passable)
    for (i, j) in exits:
        exit_mask[i, j] = True
    if not passable[start] or not (passable & exit_mask).any():
        return None
    h = distance_transform_edt(~exit_mask) * voxel / SPEEDS[WALK]
    speed = SPEEDS[np.clip(state, 0, 4)]
    penalty = 1.0 + kappa * np.clip(unknown, 0.0, 1.0)
    g = np.full(state.shape, np.inf)
    g[start] = 0.0
    parent = {}
    closed = np.zeros(state.shape, dtype=bool)
    heap = [(h[start], 0.0, start)]
    while heap:
        f, gc, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        closed[cur] = True
        if exit_mask[cur]:
            path = [cur]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path[::-1]
        ci, cj = cur
        for di, dj, dist in NEIGH:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or closed[ni, nj] or not passable[ni, nj]:
                continue
            dh = support[ni, nj] - support[ci, cj]
            if abs(dh) > MAX_STEP:
                continue
            step = (dist * voxel / speed[ni, nj] + max(0.0, dh) / V_CLIMB) * penalty[ni, nj]
            ng = gc + step
            if ng < g[ni, nj]:
                g[ni, nj] = ng
                parent[(ni, nj)] = cur
                heapq.heappush(heap, (ng + h[ni, nj], ng, (ni, nj)))
    return None


def route_summary(path, state, support, voxel=0.1):
    """Довжина ділянок за позами та час проходження (без штрафу)."""
    length = {WALK: 0.0, CROUCH: 0.0, CRAWL: 0.0}
    t = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = np.hypot(a[0] - b[0], a[1] - b[1]) * voxel
        k = int(state[b])
        length[k] += d
        t += d / SPEEDS[k] + max(0.0, support[b] - support[a]) / V_CLIMB
    return {"length_walk": length[WALK], "length_crouch": length[CROUCH], "length_crawl": length[CRAWL],
            "length_total": sum(length.values()), "time_s": t, "cells": len(path)}


def check_route(path, truth_state, truth_support, voxel=0.1):
    """Перевірка побудованого маршруту на точній моделі сцени: усі клітинки
    прохідні, уступи не вищі за MAX_STEP. Повертає (безпечний, реальний час)."""
    if path is None:
        return False, None
    t = 0.0
    for a, b in zip(path[:-1], path[1:]):
        if truth_state[b] > CRAWL or truth_state[a] > CRAWL:
            return False, None
        dh = truth_support[b] - truth_support[a]
        if abs(dh) > MAX_STEP:
            return False, None
        d = np.hypot(a[0] - b[0], a[1] - b[1]) * voxel
        t += d / SPEEDS[int(truth_state[b])] + max(0.0, dh) / V_CLIMB
    return True, t


def classify_outcome(path, safe, truth_exists):
    """Результат за класифікацією: успіх, небезпечний маршрут, пропуск, коректна відмова."""
    if path is not None and safe:
        return "success"
    if path is not None and not safe:
        return "dangerous"
    if path is None and truth_exists:
        return "miss"
    return "correct_refusal"
