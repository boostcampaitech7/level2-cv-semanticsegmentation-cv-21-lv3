import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import *

class BCEWithDICE(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(mode="multilabel")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)  # Corrected reference
        dice_loss = self.dice(inputs, targets)  # Corrected reference
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class LossSelector:
    def __init__(self, loss_config):
        """s
        Initialize LossSelector with configuration.
        :param loss_config: dict containing loss type and its parameters.
        """
        self.loss_config = loss_config
        self.loss_type = loss_config.get('type', '').lower()
        self.loss_params = loss_config.get('params', {})

    def get_loss(self):
        """
        Get the loss function based on the type specified in the config.
        """
        # 필수 파라미터만 필터링
        if self.loss_type == 'bcewithlogitsloss':
            return self._get_bcewithlogitsloss()
        elif self.loss_type == 'diceloss':
            return self._get_dice_loss()
        elif self.loss_type == 'bcewithdice':
            return self._get_bcewithdice_loss()
        else:
            raise ValueError(f"지원하지 않는 손실함수 타입입니다: {self.loss_type}")

    def _get_bcewithlogitsloss(self):
        """
        BCEWithLogitsLoss: 기본 Loss 함수
        """
        # BCEWithLogitsLoss에는 특별히 필요한 파라미터가 없으므로 그대로 넘겨줍니다.
        return nn.BCEWithLogitsLoss()

    def _get_dice_loss(self):
        """
        DiceLoss: 1 - dice coefficient
        """
        # DiceLoss에 필요한 파라미터들만 필터링
        return DiceLoss(mode="multilabel")

    def _get_bcewithdice_loss(self):
        """
        Combined BCEWithLogitsLoss and DiceLoss
        """
        d_w = self.loss_params.get('dice_weight', 0.5)
        b_w = self.loss_params.get('bce_weight', 0.5)
        return BCEWithDICE(dice_weight=d_w, bce_weight=b_w)