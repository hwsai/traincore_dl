# traincore_dl/logger_utils.py
"""
Training Logger Module
----------------------
統一管理訓練過程中的紀錄、模型、摘要與損失圖表。
適用於 CNN、GNN、Transformer 等各種架構。

包含：
- TrainingLogger：訓練全過程的統一記錄與輸出類別
  - save_model() 儲存模型
  - save_summary() 儲存訓練摘要
  - save_losses() 輸出 CSV
  - plot_losses() 即時繪圖
  - plot_from_csv() 從 CSV 重繪
"""

import os
import csv
import math
import torch
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# 子類：負責繪圖
# =====================================================
class _LossPlotter:
    def __init__(self, title="Training Loss", figsize=(8, 6), dpi=120):
        self.title = title
        self.figsize = figsize
        self.dpi = dpi

    @staticmethod
    def _nice_ceil(x, base=1, pad=1.15):
        if x is None or x <= 0:
            return base
        return base * math.ceil((x * pad) / base)

    def _draw_plot(self, train_losses, val_losses, save_path,
                   test_loss=None, training_time=None, y_max=None, log_scale=False):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.title(self.title, fontsize=12, weight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss (MSE)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        ax = plt.gca()

        if log_scale:
            plt.yscale("log")
            plt.ylim(1e-4, 1)
        else:
            if y_max is None:
                combined = np.concatenate([train_losses, val_losses])
                high_val = np.percentile(combined, 95)
                y_max = self._nice_ceil(high_val)
            plt.ylim(0, y_max)

        if training_time is not None:
            ax.text(0.99, 0.85, f"Training Time: {training_time:.2f}s",
                    ha='right', va='bottom', transform=ax.transAxes,
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
        if test_loss is not None:
            ax.text(0.99, 0.93, f"Test MSE: {test_loss:.4f}",
                    ha='right', va='bottom', transform=ax.transAxes,
                    fontsize=9, bbox=dict(facecolor='lightyellow', alpha=0.7, edgecolor='orange'))

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ Loss plot saved: {save_path}")

    def dual(self, train_losses, val_losses, base_save_path, test_loss=None, training_time=None, y_max=None):
        """輸出線性與對數兩張圖"""
        self._draw_plot(train_losses, val_losses, f"{base_save_path}_linear.png",
                        test_loss, training_time, y_max, log_scale=False)
        self._draw_plot(train_losses, val_losses, f"{base_save_path}_log.png",
                        test_loss, training_time, y_max, log_scale=True)


# =====================================================
# 主類別：整合模型紀錄與繪圖
# =====================================================
class TrainingLogger:
    def __init__(self, save_dir, model_name="model", title="Training Loss"):
        self.save_dir = save_dir
        self.model_name = model_name
        self.plotter = _LossPlotter(title=title)
        os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------
    # 模型儲存
    # ---------------------------------------------
    def save_model(self, model):
        path = os.path.join(self.save_dir, f"{self.model_name}.pth")
        torch.save(model.state_dict(), path)
        print(f"✅ Model saved: {path}")
        return path

    # ---------------------------------------------
    # 紀錄 Loss CSV
    # ---------------------------------------------
    def save_losses(self, train_losses, val_losses):
        csv_path = os.path.join(self.save_dir, f"{self.model_name}_loss.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train", "val"])
            for i, (tr, va) in enumerate(zip(train_losses, val_losses)):
                writer.writerow([i, tr, va])
        print(f"✅ Loss CSV saved: {csv_path}")
        return csv_path

    # ---------------------------------------------
    # 輸出訓練摘要
    # ---------------------------------------------
    def save_summary(self, cfg: dict, test_result=None, train_time=None):
        txt_path = os.path.join(self.save_dir, f"{self.model_name}_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            print("=== Training Summary ===", file=f)
            print(f"Model Name        : {self.model_name}", file=f)
            for k, v in cfg.items():
                print(f"{k:18s}: {v}", file=f)
            if test_result is not None:
                print(f"Final Test MSE    : {test_result:.6f}", file=f)
            if train_time is not None:
                print(f"Total Time (sec)  : {train_time:.2f}", file=f)
        print(f"✅ Summary saved: {txt_path}")
        return txt_path

    # ---------------------------------------------
    # 即時繪圖
    # ---------------------------------------------
    def plot_losses(self, train_losses, val_losses, test_loss=None, training_time=None):
        base_path = os.path.join(self.save_dir, f"{self.model_name}_loss")
        self.plotter.dual(train_losses, val_losses, base_path, test_loss=test_loss, training_time=training_time)

    # ---------------------------------------------
    # 從 CSV 重繪
    # ---------------------------------------------
    def plot_from_csv(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(self.save_dir, f"{self.model_name}_loss.csv")
        train_losses, val_losses = [], []
        with open(csv_path, "r", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                try:
                    train_losses.append(float(row[1]))
                    val_losses.append(float(row[2]))
                except ValueError:
                    continue
        self.plot_losses(train_losses, val_losses)
        print(f"✅ Replotted from CSV: {csv_path}")


# =====================================================
# CLI 模式：命令列重繪
# =====================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python logger_utils.py <loss_csv_path>")
        sys.exit(0)
    csv_path = sys.argv[1]
    logger = TrainingLogger(save_dir=os.path.dirname(csv_path))
    logger.plot_from_csv(csv_path)
