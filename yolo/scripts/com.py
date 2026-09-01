from pathlib import Path
import matplotlib.pyplot as plt

# --------------------
# CONFIG
# --------------------
GT_DIR = Path("data/plug_dataset/plug_data/labels/val")

PRED_DIRS = {
    "v8_50": Path("runs/detect/runs/detect/pred_v8_50/labels"),
    "v8_add_50": Path("runs/detect/runs/detect/pred_v8_add_50/labels"),
    "v8_add_100": Path("runs/detect/runs/detect/pred_v8_add_100/labels"),
}
IOU_TH = 0.5
CONF_TH = 0.25   # 固定阈值用于论文展示
CLS_ID = 0        # 0: plug

# --------------------
# UTILS
# --------------------
def yolo_to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def read_gt_boxes(txt_path: Path):
    if not txt_path.exists():
        return []
    out = []
    for ln in txt_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        cls = int(float(p[0]))
        if cls != CLS_ID:
            continue
        cx, cy, w, h = map(float, p[1:5])
        out.append(yolo_to_xyxy(cx, cy, w, h))
    return out

def read_pred_boxes(txt_path: Path):
    # Pred YOLO (save_conf=True): cls cx cy w h conf
    if not txt_path.exists():
        return []
    out = []
    for ln in txt_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        cls = int(float(p[0]))
        if cls != CLS_ID:
            continue
        cx, cy, w, h = map(float, p[1:5])
        conf = float(p[5]) if len(p) >= 6 else 1.0
        if conf < CONF_TH:
            continue
        out.append((yolo_to_xyxy(cx, cy, w, h), conf))
    out.sort(key=lambda x: x[1], reverse=True)  # 置信度从高到低
    return out

def match_tp_fp_fn(gt_boxes, pred_boxes):
    """
    贪心匹配：按预测置信度从高到低，给每个 pred 找 IoU 最大的未匹配 GT。
    IoU>=阈值则 TP，并占用该 GT；否则 FP。
    未匹配 GT 计为 FN。
    success: fn==0 表示该图所有 GT 都被找到了（工程意义：这张图“任务成功”）
    """
    matched_gt = [False] * len(gt_boxes)
    tp = fp = 0

    for pbox, _conf in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            v = iou(pbox, gbox)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= IOU_TH and best_j >= 0:
            tp += 1
            matched_gt[best_j] = True
        else:
            fp += 1

    fn = matched_gt.count(False)
    return tp, fp, fn, (fn == 0)

# --------------------
# MAIN
# --------------------
def main():
    gt_files = sorted(GT_DIR.glob("*.txt"))
    if not gt_files:
        raise RuntimeError(f"No GT txt found in {GT_DIR}")

    # 统计每个模型的 P/R/Success
    results = {}
    for model_name, pred_dir in PRED_DIRS.items():
        TP = FP = FN = 0
        img_success = 0
        total_imgs = 0

        for gt_path in gt_files:
            stem = gt_path.stem
            pred_path = pred_dir / f"{stem}.txt"

            gt_boxes = read_gt_boxes(gt_path)
            if len(gt_boxes) == 0:
                continue

            pred_boxes = read_pred_boxes(pred_path)
            tp, fp, fn, success = match_tp_fp_fn(gt_boxes, pred_boxes)

            TP += tp
            FP += fp
            FN += fn
            img_success += 1 if success else 0
            total_imgs += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        success_rate = img_success / total_imgs if total_imgs > 0 else 0.0

        results[model_name] = dict(
            precision=precision,
            recall=recall,
            success_rate=success_rate,
            TP=TP, FP=FP, FN=FN, total_imgs=total_imgs
        )

        print(f"\n[{model_name}]")
        print(f"TP={TP}, FP={FP}, FN={FN}, images={total_imgs}")
        print(f"Precision={precision:.3f}, Recall={recall:.3f}, Per-image Success={success_rate:.3f}")

    out_dir = Path("runs/detect/compare_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------
    # ONE FIGURE: Precision/Recall/Success (grouped bars)
    # --------------------
    names = list(results.keys())
    P = [results[n]["precision"] for n in names]
    R = [results[n]["recall"] for n in names]
    S = [results[n]["success_rate"] for n in names]

    x = list(range(len(names)))
    width = 0.25

    plt.figure(figsize=(8.5, 4.5))
    plt.bar([i - width for i in x], P, width=width, label="Precision")
    plt.bar(x, R, width=width, label="Recall")
    plt.bar([i + width for i in x], S, width=width, label="Per-image Success")

    plt.xticks(x, names)
    plt.ylim(0, 1.0)
    plt.ylabel(f"Score (IoU≥{IOU_TH}, conf≥{CONF_TH})")
    plt.title("Detection & Task-level Performance on Same Set")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "bar_P_R_S_success.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
