from pathlib import Path
import matplotlib.pyplot as plt

# ================= CONFIG =================
GT_DIR = Path("data/plug_dataset/plug_data/labels/val")

PRED_DIRS = {
    "v8_50": Path("runs/detect/runs/detect/pred_v8_50/labels"),
    "v8_add_50": Path("runs/detect/runs/detect/pred_v8_add_50/labels"),
    "v8_add_100": Path("runs/detect/runs/detect/pred_v8_add_100/labels"),
}

CONF_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
IOU_TH = 0.5
CLS_ID = 0

OUT_DIR = Path("runs/detect/compare_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================= UTILS =================
def yolo_to_xyxy(cx, cy, w, h):
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

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

def read_gt_boxes(path: Path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        if int(float(p[0])) != CLS_ID:
            continue
        cx, cy, w, h = map(float, p[1:5])
        out.append(yolo_to_xyxy(cx, cy, w, h))
    return out

def read_pred_boxes(path: Path):
    """Pred YOLO with conf: cls cx cy w h conf"""
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        if int(float(p[0])) != CLS_ID:
            continue
        cx, cy, w, h = map(float, p[1:5])
        conf = float(p[5]) if len(p) >= 6 else 1.0
        out.append((yolo_to_xyxy(cx, cy, w, h), conf))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def match_tp_fp_fn(gt_boxes, preds_filtered):
    matched = [False] * len(gt_boxes)
    tp = fp = 0
    for pbox, _conf in preds_filtered:
        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_boxes):
            if matched[j]:
                continue
            v = iou(pbox, gbox)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= IOU_TH and best_j >= 0:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1
    fn = matched.count(False)
    return tp, fp, fn

def f1(p, r):
    return (2*p*r/(p+r)) if (p+r) > 0 else 0.0


# ================= EVAL =================
def eval_f1_curve(pred_dir: Path):
    gt_files = sorted(GT_DIR.glob("*.txt"))
    data = []
    for gt_path in gt_files:
        gt = read_gt_boxes(gt_path)
        if len(gt) == 0:
            continue  # 默认跳过无 plug 的图片
        pred = read_pred_boxes(pred_dir / gt_path.name)
        data.append((gt, pred))

    f1_curve = []
    pr_curve = []  # (P,R) 仅用于找最优点，不画也行

    for conf_th in CONF_LIST:
        TP = FP = FN = 0
        for gt_boxes, pred_all in data:
            preds = [(b,c) for (b,c) in pred_all if c >= conf_th]
            tp, fp, fn = match_tp_fp_fn(gt_boxes, preds)
            TP += tp
            FP += fp
            FN += fn

        P = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        R = TP/(TP+FN) if (TP+FN) > 0 else 0.0
        F1 = f1(P, R)
        f1_curve.append(F1)
        pr_curve.append((P, R))

    return f1_curve, pr_curve


# ================= PLOT =================
def main():
    plt.figure(figsize=(7.5, 4.5))

    for name, pred_dir in PRED_DIRS.items():
        f1_curve, pr_curve = eval_f1_curve(pred_dir)

        # 找到最佳 operating point（F1 最大）
        best_idx = max(range(len(CONF_LIST)), key=lambda i: f1_curve[i])
        best_conf = CONF_LIST[best_idx]
        best_f1 = f1_curve[best_idx]
        best_p, best_r = pr_curve[best_idx]

        # 画曲线
        plt.plot(CONF_LIST, f1_curve, marker="o", label=f"{name}")

        # 标最佳点
        line = plt.gca().lines[-1]
        plt.scatter([best_conf], [best_f1], s=70, color=line.get_color(), zorder=5)
        offset_map = {
            "v8_50": (12, -10),
            "v8_add_50": (12, 10),
            "v8_add_100": (12, -28),
        }
        dx, dy = offset_map.get(name, (12, -10))

        plt.annotate(
            f"{name}\nconf={best_conf:.2f}\nF1={best_f1:.2f}\nP={best_p:.2f}, R={best_r:.2f}",
            (best_conf, best_f1),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
        )

        print(f"[{name}] best conf={best_conf:.2f}, F1={best_f1:.3f}, P={best_p:.3f}, R={best_r:.3f}")

    plt.xlabel("Confidence threshold")
    plt.ylabel(f"F1 score (IoU≥{IOU_TH})")
    plt.ylim(0, 1.0)
    plt.title("F1 vs Confidence Threshold (Same Test Set)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "curve_f1_vs_conf.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
