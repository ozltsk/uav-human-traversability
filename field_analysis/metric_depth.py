"""Метрична глибина Depth Anything V2 Metric Indoor Small (потрібен transformers >= 4.45: venv_metric).

  venv_metric/Scripts/python field_analysis/metric_depth.py check <png> [--hfov 55.5] [--tilt 10] [--tof-m 0.75]
      -- глибина в точці горизонтального променя (куди дивиться передній ToF) проти його показу;
  venv_metric/Scripts/python field_analysis/metric_depth.py log <журнал АС Р1> [...]
      -- для кожного кадру пише pred/pred_NNNN.npz: metric_depth (float16, м) поряд із наявними масивами.
"""
import argparse
import os
import sys
import time

import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def load():
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    return torch, proc, model


def predict(torch, proc, model, pil):
    with torch.no_grad():
        inp = proc(images=pil, return_tensors="pt")
        pred = model(**inp).predicted_depth[0]
        pred = torch.nn.functional.interpolate(pred[None, None], size=(pil.height, pil.width), mode="bilinear",
                                               align_corners=False)[0, 0]
    return pred.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "log"))
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--hfov", type=float, default=55.5)
    ap.add_argument("--tilt", type=float, default=10.0)
    ap.add_argument("--tof-m", type=float, default=None)
    a = ap.parse_args()
    from PIL import Image
    torch, proc, model = load()
    if a.mode == "check":
        for p in a.paths:
            pil = Image.open(p).convert("RGB")
            Z = predict(torch, proc, model, pil)
            w, h = pil.width, pil.height
            f = w / (2 * np.tan(np.radians(a.hfov) / 2))
            v = int(round(h / 2 - f * np.tan(np.radians(a.tilt))))
            win = Z[max(0, v - 12):v + 13, w // 2 - 12:w // 2 + 13]
            D = float(np.median(win)) / np.cos(np.radians(a.tilt))
            print(f"{os.path.basename(p)}: метрична глибина по променю ToF {D:.2f} м"
                  + (f" (ToF {a.tof_m:.2f} м, відношення {D / a.tof_m:.2f})" if a.tof_m else "")
                  + f"; медіана кадру {np.median(Z):.2f} м, мін {Z.min():.2f}, макс {Z.max():.2f}")
        return
    for root in a.paths:
        rows = open(os.path.join(root, "telemetry.csv"), encoding="utf-8").read().splitlines()[1:]
        t0, n = time.time(), 0
        for r in rows:
            i = int(r.split(",")[0])
            fp = os.path.join(root, "frames", f"frame_{i:04d}.png")
            pp_ = os.path.join(root, "pred", f"pred_{i:04d}.npz")
            if not os.path.exists(fp) or not os.path.exists(pp_):
                continue
            z = dict(np.load(pp_))
            if "metric_depth" in z:
                continue
            z["metric_depth"] = predict(torch, proc, model, Image.open(fp).convert("RGB")).astype(np.float16)
            np.savez_compressed(pp_, **z)
            n += 1
        print(f"{os.path.basename(root)}: метрична глибина для {n} кадрів, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
