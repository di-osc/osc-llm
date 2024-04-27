from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math

def get_cosine_lr_scheduler(optimizer: Optimizer,
                          num_warm_up_steps: int,
                          num_training_steps: int,
                          min_lr_ratio: float = 0.0,
                          num_cycles: float = 0.5):
    """线性预热与余弦衰减的学习率策略,最大学习率为优化器初始学习率,最小学习率为初始学习率的min_lr_ratio倍.

    参数:
    - optimizer (Optimizer): 优化器.
    - num_warm_up_steps (int): 学习率线性预热的步数.
    - num_training_steps (int): 训练步数.
    - min_lr_ratio (float, optional): 最小学习率与初始学习率的比例. Defaults to 0.0.
    - num_cycles (float, optional): 余弦衰减的周期数. Defaults to 0.5.
    """
    assert num_warm_up_steps < num_training_steps
    assert 0.0 <= min_lr_ratio <= 1.0
    def lr_lambda(current_step):
        if current_step < num_warm_up_steps:
            return float(current_step) / float(max(1, num_warm_up_steps))
        progress = float(current_step - num_warm_up_steps) / float(max(1, num_training_steps - num_warm_up_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress)))
    
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=-1)