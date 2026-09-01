from pathlib import Path
import yaml
from ultralytics import YOLO
model = YOLO ('/home/xykb1/yolo_ws/wire_plug_detect/.venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/myyolov8.yaml')
ROOT = Path(__file__).resolve().parents[1]

def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    data_yaml = str(ROOT / "configs" / "dataset.yaml")

    model = YOLO(cfg["model"])
    model.train(
        data=data_yaml,
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg["workers"],
        patience=cfg["patience"],
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=cfg["exist_ok"],
    )

if __name__ == "__main__":
    main()
