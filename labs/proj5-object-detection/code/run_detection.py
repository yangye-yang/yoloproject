# -*- coding: utf-8 -*-
"""
Proj5: Object Detection with open-source APIs.
Detectors: Faster R-CNN, FCOS, RetinaNet, SSDLite (torchvision, COCO)
           + YOLOv8 (ultralytics, COCO)
Runs on GPU (cuda) when available, otherwise CPU.

For every input image in ../images, each detector is run; boxes are drawn and saved.
Outputs go to ../results:
  - <detector>/<image>.jpg      per-detector annotated image
  - compare_<image>.jpg         1x5 side-by-side comparison
  - summary.json                per (image, detector) detection counts / classes / timing
  - stats.png                   detection-count bar chart per detector
"""
import os, sys, json, time, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

torch.set_num_threads(max(1, os.cpu_count() or 4))

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "..", "images")
OUT_DIR = os.path.join(HERE, "..", "results")
YOLO_CKPT = os.path.join(HERE, "..", "yolov8n.pt")
SCORE_THR = 0.5          # detection confidence threshold
MAX_SIDE = 1000          # downscale long side for CPU speed

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 20-ish distinct colors for classes
_CMAP = plt.get_cmap("tab20")


class TVDetector:
    """torchvision COCO detector wrapper (Faster R-CNN / FCOS / RetinaNet / SSDLite)."""

    def __init__(self, name, fn, weights):
        self.name = name
        self.model = fn(weights=weights).eval().to(DEVICE)
        self.classes = weights.meta["categories"]

    def run(self, img_np, repeats=3):
        t = torch.from_numpy(np.ascontiguousarray(img_np))
        t = t.permute(2, 0, 1).contiguous().float().div(255.0).to(DEVICE)
        best_dt, out = None, None
        for _ in range(repeats):
            t0 = time.time()
            with torch.no_grad():
                out = self.model([t])[0]
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            best_dt = dt if best_dt is None else min(best_dt, dt)
        keep = out["scores"] >= SCORE_THR
        return (out["boxes"][keep].cpu().numpy(),
                out["labels"][keep].cpu().numpy(),
                out["scores"][keep].cpu().numpy(), best_dt)


class YoloDetector:
    """ultralytics YOLOv8 COCO detector wrapper."""

    def __init__(self, name, ckpt):
        from ultralytics import YOLO
        self.name = name
        self.model = YOLO(ckpt)
        self.classes = [self.model.names[i] for i in sorted(self.model.names)]

    def run(self, img_np, repeats=3):
        dev = "cuda" if DEVICE.type == "cuda" else "cpu"
        best_dt, r = None, None
        for _ in range(repeats):
            t0 = time.time()
            r = self.model.predict(Image.fromarray(img_np), conf=SCORE_THR,
                                   device=dev, verbose=False)[0]
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            best_dt = dt if best_dt is None else min(best_dt, dt)
        b = r.boxes
        return (b.xyxy.cpu().numpy(),
                b.cls.cpu().numpy().astype(int),
                b.conf.cpu().numpy(), best_dt)


def build_models():
    torchvision_specs = [
        ("FasterRCNN", fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
        ("FCOS",       fcos_resnet50_fpn,       FCOS_ResNet50_FPN_Weights.DEFAULT),
        ("RetinaNet",  retinanet_resnet50_fpn,  RetinaNet_ResNet50_FPN_Weights.DEFAULT),
        ("SSDLite",    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights.DEFAULT),
    ]
    models = []
    for name, fn, weights in torchvision_specs:
        t0 = time.time()
        det = TVDetector(name, fn, weights)
        models.append(det)
        print("[load] %-11s ready in %.1fs  (%d classes)" % (name, time.time() - t0, len(det.classes)))
    if os.path.exists(YOLO_CKPT):
        t0 = time.time()
        det = YoloDetector("YOLOv8", YOLO_CKPT)
        models.append(det)
        print("[load] %-11s ready in %.1fs  (%d classes)" % ("YOLOv8", time.time() - t0, len(det.classes)))
    else:
        print("[skip] YOLOv8: weights not found at", YOLO_CKPT)
    return models


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.asarray(img)


def draw(ax, img_np, boxes, labels, scores, classes):
    ax.imshow(img_np)
    ax.axis("off")
    used = sorted(set(int(l) for l in labels))
    color_of = {c: _CMAP(i % 20) for i, c in enumerate(used)}
    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        c = color_of[int(lab)]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=2))
        ax.text(x1, max(y1 - 4, 8), "%s %.2f" % (classes[int(lab)], sc),
                color="white", fontsize=8, va="bottom",
                bbox=dict(fc=c, ec="none", pad=1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
                       glob.glob(os.path.join(IMG_DIR, "*.png")) +
                       glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
    if not img_paths:
        print("no images found in", IMG_DIR); sys.exit(1)
    print("images:", [os.path.basename(p) for p in img_paths])
    print("device:", DEVICE)

    models = build_models()
    n_det = len(models)

    # warm up CUDA / cuDNN so first-image timings are not distorted
    dummy = np.zeros((600, 800, 3), dtype=np.uint8)
    for det in models:
        det.run(dummy, repeats=1)
    print("warmup done")

    summary = {}
    for path in img_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            img_np = load_image(path)
        except Exception as e:
            print("  [skip] %s -> %s" % (base, e)); continue
        fig, axes = plt.subplots(1, n_det, figsize=(6 * n_det, 5))
        if n_det == 1:
            axes = [axes]
        summary[base] = {}
        for ax, det in zip(axes, models):
            boxes, labels, scores, dt = det.run(img_np)
            d = os.path.join(OUT_DIR, det.name); os.makedirs(d, exist_ok=True)
            # per-detector standalone image
            f2, a2 = plt.subplots(figsize=(7, 6))
            draw(a2, img_np, boxes, labels, scores, det.classes)
            a2.set_title("%s  (%d detections, %.1fs)" % (det.name, len(boxes), dt))
            f2.tight_layout(); f2.savefig(os.path.join(d, base + ".jpg"), dpi=110); plt.close(f2)
            # panel in comparison grid
            draw(ax, img_np, boxes, labels, scores, det.classes)
            ax.set_title("%s  (%d, %.1fs)" % (det.name, len(boxes), dt))
            cls_names = [det.classes[int(l)] for l in labels]
            summary[base][det.name] = {
                "n": int(len(boxes)),
                "time_s": round(dt, 2),
                "classes": {c: cls_names.count(c) for c in sorted(set(cls_names))},
                "scores": [round(float(s), 3) for s in scores],
            }
            print("  %-11s %-28s n=%-3d %.2fs" % (det.name, base, len(boxes), dt))
        fig.suptitle("Comparison on %s" % base, fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "compare_%s.jpg" % base), dpi=110)
        plt.close(fig)

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # stats bar chart: total detections per detector
    names = [m.name for m in models]
    totals = [sum(summary[b][n]["n"] for b in summary) for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = ["#2e6fdb", "#e0662f", "#3aa65a", "#b34fb0", "#c9a227"]
    ax.bar(names, totals, color=palette[:len(names)])
    ax.set_ylabel("total detections (score>=%.1f)" % SCORE_THR)
    ax.set_title("Detections per detector over %d images (%s)" % (len(summary), DEVICE))
    for i, v in enumerate(totals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "stats.png"), dpi=120); plt.close(fig)
    print("done -> ", OUT_DIR)


if __name__ == "__main__":
    main()
