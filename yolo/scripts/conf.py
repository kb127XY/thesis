from ultralytics import YOLO
import matplotlib.pyplot as plt

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")  # 或你的自训模型

# 推理：conf 设为 0，拿到“所有候选框”
results = model("/mnt/e/thesis/image/top/all/frame1 (30).png", conf=0.0, verbose=False)

# 取出所有 confidence
confidences = results[0].boxes.conf.cpu().numpy()

# 作图
plt.figure(figsize=(6, 4))
plt.hist(confidences, bins=20, edgecolor="black")

# 画 conf 阈值线
plt.axvline(0.25, color="red", linestyle="--", label="conf = 0.25")
plt.axvline(0.18, color="blue", linestyle="--", label="conf = 0.18")

plt.xlabel("Confidence")
plt.ylabel("Number of boxes")
plt.title("Confidence distribution for a single image")
plt.legend()
plt.tight_layout()
plt.savefig("conf_distribution.png", dpi=200)

