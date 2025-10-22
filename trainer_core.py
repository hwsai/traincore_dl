# traincore_dl/trainer_core.py
"""
Trainer Core
------------
通用的訓練與驗證流程模組，可適用於 CNN、GNN 或混合架構。
包含：
- 訓練主迴圈 (train_one_epoch, validate_one_epoch)
- EarlyStopping 機制
- 統一的 run() 接口
"""

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device):
        # 初始化模型、優化器、學習率排程器等
        pass

    def train_one_epoch(self, loader):
        # 執行一個 epoch 的訓練
        pass

    def validate_one_epoch(self, loader):
        # 執行驗證
        pass

    def run(self, train_loader, val_loader, epochs):
        # 主訓練流程：控制 epoch、scheduler、early stopping
        pass