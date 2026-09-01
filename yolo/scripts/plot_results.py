import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 多个实验 results.csv 路径（按你的实际目录改）
CSV_PATHS = {
    "v8_50": "runs/detect/train/results.csv",
    "v8_add_50": "runs/detect/train3/results.csv",
    "v8_add_100": "runs/detect/train4/results.csv",
}

OUT_DIR = Path("runs/detect/compare_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_metric(metric_key: str, title: str):
    plt.figure(figsize=(10, 6))

    plotted_any = False
    for label, csv_path in CSV_PATHS.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"[WARN] Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        x = df["epoch"] if "epoch" in df.columns else range(len(df))

        if metric_key in df.columns:
            plt.plot(x, df[metric_key], label=label)
            plotted_any = True
        else:
            print(f"[WARN] Column '{metric_key}' not found in {csv_path}")

    if not plotted_any:
        plt.close()
        print(f"[SKIP] Nothing plotted for {title}")
        return

    plt.xlabel("epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / f"compare_{title}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {out_path}")

def main():
    plot_metric("metrics/mAP50(B)", "mAP50")
    plot_metric("metrics/mAP50-95(B)", "mAP50-95")
    plot_metric("metrics/precision(B)", "Precision")
    plot_metric("metrics/recall(B)", "Recall")
    plot_metric("train/box_loss", "train_box_loss")
    plot_metric("val/box_loss", "val_box_loss")

if __name__ == "__main__":
    main()