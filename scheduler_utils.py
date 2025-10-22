# traincore_dl/scheduler_utils.py
"""
Scheduler Utilities
-------------------
學習率控制模組：
- build_scheduler(): 建立 step decay, cosine decay, 或固定 LR
- NoOpScheduler: 無動作的替代方案
- WarmupLR: 預熱學習率
"""

def build_scheduler(optimizer, scheme="step50_75"):
    pass

class NoOpScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self):
        pass