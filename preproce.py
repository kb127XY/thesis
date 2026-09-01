import os
import cv2
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_ROOT = os.path.join(BASE_DIR, "top")
OUTPUT_ROOT = os.path.join(BASE_DIR, "top_crop")
# =============================================================

IMG_EXT = (".png", ".jpg", ".jpeg")

# 640x480 下的“桌面范围”（先给一个通用默认值；你可按实际略微调整）
# 解释：只要把明显不是桌面的上边/两侧裁掉即可，别裁太小
DESK_ROI = dict(x1=0, y1=24, x2=639, y2=400)

# 自动 bbox 的 padding（像素），宁可大一些
PADDING = 25

# 最小前景面积阈值（过滤噪声）
MIN_AREA = 800

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2

def auto_bbox_in_desk(img_bgr):
    """在 Desk ROI 内自动找前景 bbox。失败返回 None。"""
    h, w = img_bgr.shape[:2]
    dx1, dy1, dx2, dy2 = DESK_ROI["x1"], DESK_ROI["y1"], DESK_ROI["x2"], DESK_ROI["y2"]
    desk = img_bgr[dy1:dy2, dx1:dx2]

    # 灰度 + 轻微平滑，抗噪
    gray = cv2.cvtColor(desk, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 用 Otsu 自动阈值：把“更暗/更复杂”的区域当作前景
    # 注意：桌面通常更亮更均匀，因此这里用 THRESH_BINARY_INV 更常见
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学：去小噪声、连通碎片
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 找连通域 / 轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 选最大面积的前景（你也可以改成“最靠近桌面中心”的策略）
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if area > best_area:
            best_area = area
            best = (x, y, x + ww, y + hh)

    if best is None:
        return None

    # 转回到全图坐标系 + padding
    x1, y1, x2, y2 = best
    x1 = x1 + dx1 - PADDING
    y1 = y1 + dy1 - PADDING
    x2 = x2 + dx1 + PADDING
    y2 = y2 + dy1 + PADDING

    return clamp_box(x1, y1, x2, y2, w, h)

def process_one_folder(wire_dir, out_dir, wire_name):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(wire_dir) if f.lower().endswith(IMG_EXT)])

    for idx, fname in enumerate(files):
        path = os.path.join(wire_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 读图失败: {path}")
            continue

        bbox = auto_bbox_in_desk(img)

        if bbox is None:
            # 失败回退：用 desk ROI 裁剪，保证不崩
            x1, y1, x2, y2 = DESK_ROI["x1"], DESK_ROI["y1"], DESK_ROI["x2"], DESK_ROI["y2"]
        else:
            x1, y1, x2, y2 = bbox

        crop = img[y1:y2, x1:x2]
        out_name = f"{wire_name}_{idx:04d}.png"
        cv2.imwrite(os.path.join(out_dir, out_name), crop)

    print(f"[DONE] {wire_name}: {len(files)} 张")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for wire_name in sorted(os.listdir(INPUT_ROOT)):
        wire_dir = os.path.join(INPUT_ROOT, wire_name)
        if not os.path.isdir(wire_dir):
            continue

        out_dir = os.path.join(OUTPUT_ROOT, wire_name)
        print(f"[INFO] {wire_name} ...")
        process_one_folder(wire_dir, out_dir, wire_name)

    print("=== All done ===")

if __name__ == "__main__":
    main()
