"""Повна семантична розмітка кадрів журналу АС Р1 (SegFormer ADE20K, 150 класів) -> масив labels у pred/*.npz.

  python add_labels.py <лог> [<лог> ...]      (пропускає кадри, у яких labels уже є)

Потрібна для шару перешкод за контактом з підлогою (contact_layer.py): маска підлоги АС Р1 лишається
як була, додається лише масив labels (uint8, H x W).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ac1_evacmap"))
from flight_log import FlightLog      # noqa: E402
from perception import SEG_MODEL      # noqa: E402


def main():
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(SEG_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    for root in sys.argv[1:]:
        log = FlightLog.load(root)
        done = 0
        t0 = time.time()
        for rec in log.records:
            p = log.pred_path(rec.index)
            if not os.path.exists(p) or not log.has_frame(rec.index):
                continue
            z = dict(np.load(p))
            if "labels" in z:
                continue
            from PIL import Image
            pil = Image.open(log.frame_path(rec.index)).convert("RGB")
            with torch.no_grad():
                inp = proc(images=pil, return_tensors="pt")
                logits = model(**inp).logits
                logits = torch.nn.functional.interpolate(logits, size=(pil.height, pil.width), mode="bilinear",
                                                         align_corners=False)
                z["labels"] = logits.argmax(1)[0].numpy().astype(np.uint8)
            np.savez_compressed(p, **z)
            done += 1
        print(f"{os.path.basename(root)}: розмічено {done} кадрів, {time.time() - t0:.0f} с")
    with open(os.path.join(HERE, "ade20k_labels.txt"), "w", encoding="utf-8") as f:
        for k in sorted(id2label):
            f.write(f"{k}\t{id2label[k]}\n")


if __name__ == "__main__":
    main()
