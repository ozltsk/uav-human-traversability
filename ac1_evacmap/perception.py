"""Модуль нейромережевого сприйняття кадру.

Depth Anything V2 Small повертає афінно-інваріантну обернену глибину
(більше значення -- ближче), SegFormer-B0, навчений на ADE20K, -- семантичну
маску, з якої виділяється опорна поверхня (підлога, земля, килим, дорога
та споріднені класи). Обидві моделі завантажуються з Hugging Face Hub.
"""
import time

import numpy as np

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
FLOOR_CLASSES = ("floor", "earth", "rug", "road", "sidewalk", "path", "field",
                 "sand", "grass", "ground")


class Perception:
    """Нейромережеве сприйняття: обернена глибина + маска опорної поверхні."""

    def __init__(self, device="cpu", floor_classes=FLOOR_CLASSES):
        import torch
        from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                                  SegformerForSemanticSegmentation, SegformerImageProcessor)
        self.torch = torch
        self.device = device
        self.depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
        self.seg_proc = SegformerImageProcessor.from_pretrained(SEG_MODEL)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()
        id2label = self.seg_model.config.id2label
        self.floor_ids = [int(i) for i, name in id2label.items() if name in floor_classes]
        self.last_timing = {}

    def _resize(self, arr, hw):
        t = self.torch.as_tensor(arr)[None, None].float()
        out = self.torch.nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)
        return out[0, 0].numpy()

    def predict(self, image):
        """image -- RGB-масив H x W x 3 (uint8). Повертає (rel_inv_depth, floor_mask)."""
        from PIL import Image
        torch = self.torch
        pil = Image.fromarray(np.asarray(image))
        H, W = pil.height, pil.width
        t0 = time.perf_counter()
        with torch.no_grad():
            inp = self.depth_proc(images=pil, return_tensors="pt").to(self.device)
            pred = self.depth_model(**inp).predicted_depth[0].cpu().numpy()
        rel_inv_depth = self._resize(pred, (H, W)).astype(np.float32)
        t1 = time.perf_counter()
        with torch.no_grad():
            inp = self.seg_proc(images=pil, return_tensors="pt").to(self.device)
            logits = self.seg_model(**inp).logits
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear",
                                                     align_corners=False)
            labels = logits.argmax(1)[0].cpu().numpy()
        floor_mask = np.isin(labels, self.floor_ids)
        t2 = time.perf_counter()
        self.last_timing = {"depth_s": t1 - t0, "seg_s": t2 - t1}
        return rel_inv_depth, floor_mask


class RecordedPerception:
    """Сприйняття з журналу: читає збережені передбачення pred/*.npz
    (результат симулятора або попереднього запуску нейромереж)."""

    def __init__(self, log):
        self.log = log
        self.last_timing = {"depth_s": 0.0, "seg_s": 0.0}

    def predict_index(self, index):
        return self.log.load_pred(index)


def run_perception_on_log(log, perception, overwrite=False, verbose=True):
    """Обробляє всі кадри журналу нейромережами та записує pred/*.npz."""
    done = 0
    for rec in log.records:
        if log.has_pred(rec.index) and not overwrite:
            continue
        if not log.has_frame(rec.index):
            continue
        img = log.load_frame(rec.index)
        rel, mask = perception.predict(img)
        log.save_pred(rec.index, rel, mask)
        done += 1
        if verbose:
            t = perception.last_timing
            print(f"кадр {rec.index:4d}: глибина {t['depth_s']:.2f} с, сегментація {t['seg_s']:.2f} с, "
                  f"підлога {100.0 * mask.mean():.1f} % пікселів")
    return done
