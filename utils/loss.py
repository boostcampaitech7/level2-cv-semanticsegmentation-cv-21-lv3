import torch.optim as optim
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0  # 첫 번째 주기의 길이 (epoch 단위)
        self.T_mult = T_mult  # 각 주기가 늘어나는 비율
        self.base_eta_max = eta_max  # 최대 학습률의 초기값 (변하지 않음)
        self.eta_max = eta_max  # 현재 최대 학습률 (cycle이 진행되면서 감소할 수 있음)
        self.T_up = T_up  # Warm-Up 기간
        self.T_i = T_0  # 현재 주기의 길이
        self.gamma = gamma  # 최대 학습률을 감소시키는 비율
        self.cycle = 0  # 현재 cycle(재시작 횟수)
        self.T_cur = last_epoch  # 현재 epoch 위치
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1: # 초기 상태
            return self.base_lrs
        elif self.T_cur < self.T_up: # Warm-Up 기간 (lr 선형적으로 증가)
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else: # 주기 내 학습률 감소 (cosine 함수 형태로 감소)
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    # epoch마다 학습률 업데이트
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i: 
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0: # 첫 번째 주기 이후?
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult)) # 현재 epoch가 몇 번째 cycle인지
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1) # 현재 cycle 내 위치
                    self.T_i = self.T_0 * self.T_mult ** (n) # 현재 cycle의 길이
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class LRSchedulerSelector:
    def __init__(self, optimizer, scheduler_config):
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.scheduler_type = scheduler_config.get('type', '').lower()

    def get_scheduler(self):
        if self.scheduler_type == 'lambda':
            return self._get_lambda_lr()
        elif self.scheduler_type == 'step':
            return self._get_step_lr()
        elif self.scheduler_type == 'multistep':
            return self._get_multistep_lr()
        elif self.scheduler_type == 'exp':
            return self._get_exponential_lr()
        elif self.scheduler_type == 'cosine':
            return self._get_cosine_annealing_lr()
        elif self.scheduler_type == 'cyclic':
            return self._get_cyclic_lr()
        elif self.scheduler_type == 'custom_cosine_warmup':
            return self._get_cosine_annealing_warm_up_restarts()
        else:
            raise ValueError(f"지원하지 않는 스케줄러 타입입니다: {self.scheduler_type}")

    def _get_lambda_lr(self):
        """
        사용자 정의 learning rate 조절
        lambda_func: epoch -> learning rate multiplier
        """
        lambda_func = eval(self.scheduler_config.get('lambda_func', 'lambda epoch: 1.0'))
        return optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda_func
        )

    def _get_step_lr(self):
        """
        지정된 step_size마다 gamma 비율로 learning rate 감소
        """
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.scheduler_config.get('step_size', 30),
            gamma=self.scheduler_config.get('gamma', 0.1)
        )

    def _get_multistep_lr(self):
        """
        지정된 milestones마다 gamma 비율로 learning rate 감소
        """
        return optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.scheduler_config.get('milestones', [30, 60, 90]),
            gamma=self.scheduler_config.get('gamma', 0.1)
        )

    def _get_exponential_lr(self):
        """
        매 epoch마다 gamma 비율로 exponential하게 learning rate 감소
        """
        return optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.scheduler_config.get('gamma', 0.95)
        )

    def _get_cosine_annealing_lr(self):
        """
        Cosine 함수 형태로 learning rate 조절
        """
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.scheduler_config.get('t_max', 50),
            eta_min=self.scheduler_config.get('eta_min', 0)
        )

    def _get_cyclic_lr(self):
        """
        Cyclical learning rates
        """
        return optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.scheduler_config.get('base_lr', 0.001),
            max_lr=self.scheduler_config.get('max_lr', 0.1),
            step_size_up=self.scheduler_config.get('step_size_up', 2000),
            mode=self.scheduler_config.get('mode', 'triangular'),
            cycle_momentum=self.scheduler_config.get('cycle_momentum', False)
        )

    def _get_cosine_annealing_warm_up_restarts(self):
        """
        Cosine annealing with warm up and restarts
        """
        return CosineAnnealingWarmUpRestarts(
            self.optimizer,
            T_0=self.scheduler_config.get('t_0', 50),
            T_mult=self.scheduler_config.get('t_mult', 1),
            eta_max=self.scheduler_config.get('eta_max', 0.1),
            T_up=self.scheduler_config.get('t_up', 0),
            gamma=self.scheduler_config.get('gamma', 1.0)
        ) 