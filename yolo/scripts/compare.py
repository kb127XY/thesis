import pandas as pd
import matplotlib.pyplot as plt

runs = {
    "base": "runs/detect/train/results.csv",
    "iou085": "runs/detect/train_iou085/results.csv",
}

plt.figure(figsize=(10,6))

for name, path in runs.items():
    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label=f"{name} mAP50")

plt.xlabel("Epoch")
plt.ylabel("mAP50")
plt.title("YOLO Detection Performance Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("runs/compare_map50.png")
print("Saved to runs/compare_map50.png")
