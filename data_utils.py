# traincore_dl/data_utils.py
"""
Data Utilities
--------------
共用的資料處理函式：
- set_seed(): 控制隨機種子
- inspect_dataset(): 統計 x/y 的分佈
- normalize_data(): Min-Max 或 Z-score 正規化
- scale_y(): 固定比例縮放 (for regression)
"""

def set_seed(seed: int):
    pass

def inspect_dataset(dataset):
    # 計算 x, y 的 mean, min, max
    pass

def normalize_data(tensor, mode="minmax", stats=None):
    # 支援 min-max 或 z-score 正規化
    pass

def scale_y(tensor, scale_value=100.0):
    # 固定比例縮放
    pass
