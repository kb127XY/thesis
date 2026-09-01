import os
import xml.etree.ElementTree as ET
from collections import defaultdict

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def parse_points(points_str):
    # "x1,y1;x2,y2;..."
    pts = []
    for p in points_str.strip().split(";"):
        if not p:
            continue
        x, y = p.split(",")
        pts.append((float(x), float(y)))
    return pts

def bbox_from_points(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def yolo_line(cls_id, xmin, ymin, xmax, ymax, w, h):
    xmin = clamp(xmin, 0, w - 1)
    xmax = clamp(xmax, 0, w - 1)
    ymin = clamp(ymin, 0, h - 1)
    ymax = clamp(ymax, 0, h - 1)
    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return f"{cls_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"

def main(xml_path, images_dir, out_labels_dir, classes):
    os.makedirs(out_labels_dir, exist_ok=True)

    # label -> class_id
    class_map = {name: i for i, name in enumerate(classes)}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # image_id/name -> (width,height)
    img_meta = {}
    # image_name -> list of (label, xmin,ymin,xmax,ymax)
    ann = defaultdict(list)

    for img in root.findall("image"):
        name = img.get("name")
        w = int(float(img.get("width")))
        h = int(float(img.get("height")))
        img_meta[name] = (w, h)

        # 1) box
        for box in img.findall("box"):
            label = box.get("label")
            if label not in class_map:
                continue
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            ann[name].append((label, xtl, ytl, xbr, ybr))

        # 2) polygon
        for poly in img.findall("polygon"):
            label = poly.get("label")
            if label not in class_map:
                continue
            pts = parse_points(poly.get("points"))
            xmin, ymin, xmax, ymax = bbox_from_points(pts)
            ann[name].append((label, xmin, ymin, xmax, ymax))

        # 3) polyline（有些人用 polyline 勾轮廓）
        for pl in img.findall("polyline"):
            label = pl.get("label")
            if label not in class_map:
                continue
            pts = parse_points(pl.get("points"))
            xmin, ymin, xmax, ymax = bbox_from_points(pts)
            ann[name].append((label, xmin, ymin, xmax, ymax))

    # 为每张图写 txt（即使没有标注也会写空文件，训练更稳定）
    # 注意：图片名可能带子目录，这里按文件名生成 txt；如你有重名图片，需改成保留路径结构
    for img_name, (w, h) in img_meta.items():
        base = os.path.splitext(os.path.basename(img_name))[0]
        txt_path = os.path.join(out_labels_dir, base + ".txt")

        lines = []
        for (label, xmin, ymin, xmax, ymax) in ann.get(img_name, []):
            cls_id = class_map[label]
            lines.append(yolo_line(cls_id, xmin, ymin, xmax, ymax, w, h))

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # 输出 classes 文件（可选）
    with open(os.path.join(out_labels_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    print("Done.")
    print(f"XML: {xml_path}")
    print(f"Images dir (not used for parsing, only for your reference): {images_dir}")
    print(f"Labels out: {out_labels_dir}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    # 按你的类别名改这里：如果你就一个类，比如 plug
    #classes = ["plug"]
    #多个标签
    classes = ["plug", "plug1", "plug2", "plug3", "plug4"]


    # 改成你实际路径
    xml_path = "/mnt/e/thesis/image/muti/annotations.xml"
    images_dir = "/mnt/e/thesis/image/muti/images"
    out_labels_dir = "/mnt/e/thesis/image/muti/labels_yolo"

    main(xml_path, images_dir, out_labels_dir, classes)
