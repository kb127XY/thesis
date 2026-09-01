import os
from pathlib import Path
import argparse
import cv2
import numpy as np

def yolo_xywhn_to_xyxy(img_w, img_h, xywhn):
    x, y, w, h = xywhn
    x *= img_w; y *= img_h; w *= img_w; h *= img_h
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def read_gt_labels(label_path):
    # YOLO GT: cls x y w h
    gts = []
    if not label_path.exists():
        return gts
    txt = label_path.read_text().strip()
    if not txt:
        return gts
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xywhn = list(map(float, parts[1:5]))
        gts.append((cls, xywhn))
    return gts

def read_pred_labels(pred_path):
    # YOLO pred: cls x y w h conf  (ultralytics save_txt + save_conf)
    preds = []
    if not pred_path.exists():
        return preds
    txt = pred_path.read_text().strip()
    if not txt:
        return preds
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls = int(float(parts[0]))
        xywhn = list(map(float, parts[1:5]))
        conf = float(parts[5])
        preds.append((cls, xywhn, conf))
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="/home/xykb1/yolo_ws/wire_plug_detect/data/plug_dataset/plug_data/images/val")
    ap.add_argument("--gt_labels_dir", required=True, help="/home/xykb1/yolo_ws/wire_plug_detect/data/plug_dataset/plug_data/labels/val")
    ap.add_argument("--pred_labels_dir", required=True, help="/home/xykb1/yolo_ws/wire_plug_detect/runs/detect/runs/detect/pred_v8_cbam1/labels")
    ap.add_argument("--out_dir", default="data/hardneg_from_val", help="output dataset root")
    ap.add_argument("--target_cls", type=int, default=0, help="plug1 class id")
    ap.add_argument("--conf_min", type=float, default=0.25, help="min conf to keep as hard negative")
    ap.add_argument("--iou_match", type=float, default=0.5, help="IoU threshold for matching to GT")
    ap.add_argument("--max_per_image", type=int, default=5, help="limit crops per image")
    ap.add_argument("--save_crops", action="store_true", help="save FP crops as additional negative images")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_labels_dir)
    pred_dir = Path(args.pred_labels_dir)

    out_root = Path(args.out_dir)
    out_img = out_root / "images" / "train"
    out_lbl = out_root / "labels" / "train"
    out_crop = out_root / "crops" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    out_crop.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths += list(images_dir.glob(ext))

    kept_images = 0
    kept_crops = 0

    for img_path in sorted(img_paths):
        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.txt"
        pred_path = pred_dir / f"{stem}.txt"

        gts = read_gt_labels(gt_path)
        preds = read_pred_labels(pred_path)
        if not preds:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Convert GT boxes to xyxy for matching
        gt_xyxy = []
        for cls, xywhn in gts:
            gt_xyxy.append((cls, yolo_xywhn_to_xyxy(w, h, xywhn)))

        # Find plug1 false positives
        fp_boxes = []
        for cls, xywhn, conf in preds:
            if cls != args.target_cls or conf < args.conf_min:
                continue
            pxyxy = yolo_xywhn_to_xyxy(w, h, xywhn)

            matched = False
            for gt_cls, gxyxy in gt_xyxy:
                if gt_cls != cls:
                    continue
                if iou_xyxy(pxyxy, gxyxy) >= args.iou_match:
                    matched = True
                    break
            if not matched:
                fp_boxes.append((conf, pxyxy))

        if not fp_boxes:
            continue

        # 轻量版：把整帧加入 hard negatives（label 为空）
        # 这样能让模型学会：这张“容易误报的背景图”其实无目标
        out_img_path = out_img / img_path.name
        if not out_img_path.exists():
            cv2.imwrite(str(out_img_path), img)
            (out_lbl / f"{stem}.txt").write_text("")  # empty label
            kept_images += 1

        # 强化版：把 FP 区域 crop 出来作为额外负样本（可选）
        if args.save_crops:
            fp_boxes.sort(key=lambda x: x[0], reverse=True)
            for i, (conf, b) in enumerate(fp_boxes[: args.max_per_image]):
                x1, y1, x2, y2 = b
                x1 = int(max(0, np.floor(x1))); y1 = int(max(0, np.floor(y1)))
                x2 = int(min(w, np.ceil(x2)));  y2 = int(min(h, np.ceil(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                crop_name = f"{stem}_fp{i}_c{conf:.2f}.jpg"
                cv2.imwrite(str(out_crop / crop_name), crop)
                # crop 也写一个空 label（作为纯负样本训练不行：YOLO检测默认需要原图尺度）
                # 因此 crop 更适合给你后面做“二级分类器”或“hard negative 分类过滤”用
                kept_crops += 1

    print(f"Saved hard-negative full images: {kept_images}")
    if args.save_crops:
        print(f"Saved FP crops (for analysis/extra classifier): {kept_crops}")

if __name__ == "__main__":
    main()