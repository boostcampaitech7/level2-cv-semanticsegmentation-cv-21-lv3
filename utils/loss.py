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
        elif self.loss_type == 'swinbcewithdice':  # config와 일치하도록 수정
            return self._get_swinbcewithdice_loss()
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
    
    def _get_swinbcewithdice_loss(self):
       """
       SwinV2-L optimized BCEWithDICE loss
       """
       d_w = self.loss_params.get('dice_weight', 0.5)
       b_w = self.loss_params.get('bce_weight', 0.5)
       smooth = self.loss_params.get('smooth', 1e-5)
       return SwinBCEWithDICE(
           dice_weight=d_w, 
           bce_weight=b_w,
           smooth=smooth
       )
    
    
class SwinBCEWithDICE(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        
        # DiceLoss with improved numerical stability
        self.dice = DiceLoss(
            mode="multilabel",
            smooth=self.smooth,
            log_loss=False,  # log_loss=True를 사용하면 수치 안정성이 향상되지만 학습이 느려질 수 있음
        )
        
        # BCEWithLogitsLoss with improved numerical stability
        self.bce = nn.BCEWithLogitsLoss(
            reduction='mean',
            pos_weight=None  # 클래스 불균형이 심한 경우 pos_weight 설정 고려
        )

    def forward(self, inputs, targets):
        # Ensure float32 precision
        inputs = inputs.float()
        targets = targets.float()
        
        # memory efficient forward pass
        with torch.cuda.amp.autocast(enabled=True):  # mixed precision 지원
            bce_loss = self.bce(inputs, targets)
            dice_loss = self.dice(inputs, targets)
            
            # weighted sum
            loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
            
        return loss

    def __repr__(self):
        return f"BCEWithDICE(dice_weight={self.dice_weight}, bce_weight={self.bce_weight}, smooth={self.smooth})"