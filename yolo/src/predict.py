from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    # 默认寻找 runs 下最新 best.pt
    best = None
    for p in (ROOT / "runs").rglob("best.pt"):
        best = p
        break
    if best is None:
        raise SystemExit("[ERR] best.pt not found under runs/. Please train first.")

    model = YOLO(str(best))
    source = ROOT / "data" / "images" / "val"
    model.predict(source=str(source), conf=0.25, save=True)

if __name__ == "__main__":
    main()
