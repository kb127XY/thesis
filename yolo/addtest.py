# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from tests import CFG, MODEL, MODELS, SOURCE, SOURCES_LIST, TASK_MODEL_DATA
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA, TASKS
from ultralytics.data.build import load_inference_source
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    IS_JETSON,
    IS_RASPBERRYPI,
    LINUX,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    YAML,
    checks,
    is_github_action_running,
)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_1_13
CFG = '/home/xykb1/yolo_ws/wire_plug_detect/.venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/myyolov8.yaml'
SOURCE = '/home/xykb1/yolo_ws/wire_plug_detect/data/images/train/frame_000001.png'
def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment
   

    print("Model structure:")
    print(model.model)

    total_params = sum(p.numel() for p in model.model.parameters())
    print("Total params:", total_params)
if __name__ == "__main__":
    test_model_forward()