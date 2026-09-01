import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def guess_label_path(img: Path, root: Path):
    cand = img.with_suffix(".txt")
    if cand.exists():
        return cand
    parts = img.parts
    try:
        idx = parts.index("images")
        lab = Path(*parts[:idx], "labels", *parts[idx+1:]).with_suffix(".txt")
        if lab.exists():
            return lab
    except ValueError:
        pass
    hits = list(root.rglob(img.stem + ".txt"))
    if len(hits) == 1:
        return hits[0]
    return None

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CVAT extracted folder")
    ap.add_argument("--out", default="data", help="project data dir")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    img_train = out / "images" / "train"
    img_val   = out / "images" / "val"
    lab_train = out / "labels" / "train"
    lab_val   = out / "labels" / "val"

    imgs = find_images(src)
    if not imgs:
        raise SystemExit(f"[ERR] no images found under: {src}")

    pairs, miss = [], 0
    for img in imgs:
        lab = guess_label_path(img, src)
        if lab is None:
            miss += 1
            continue
        pairs.append((img, lab))

    if not pairs:
        raise SystemExit("[ERR] no image-label pairs found.")
    print(f"[INFO] images={len(imgs)} paired={len(pairs)} missing_labels={miss}")

    for p in [img_train, img_val, lab_train, lab_val]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:] if n_train < len(pairs) else pairs[-max(1, len(pairs)//10):]

    def copy_set(pairs_list, oimg, olab):
        for img, lab in pairs_list:
            safe_copy(img, oimg / img.name)
            safe_copy(lab, olab / (Path(img.name).stem + ".txt"))

    copy_set(train_pairs, img_train, lab_train)
    copy_set(val_pairs, img_val, lab_val)

    print(f"[DONE] train={len(train_pairs)} val={len(val_pairs)}")

if __name__ == "__main__":
    main()
