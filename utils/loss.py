import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import *
import torch.nn.functional as F


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
        self.loss_config = loss_config
        self.loss_type = loss_config.get('type', '').lower()
        self.loss_params = loss_config.get('params', {})
        print(f"Initialized LossSelector with type: {self.loss_type}")  # 디버깅용

    def get_loss(self):
        print(f"Getting loss for type: {self.loss_type}")  # 디버깅용
        if self.loss_type == 'bcewithlogitsloss':
            return self._get_bcewithlogitsloss()
        elif self.loss_type == 'diceloss':
            return self._get_dice_loss()
        elif self.loss_type == 'bcewithdice':
            return self._get_bcewithdice_loss()
        elif self.loss_type == 'beitdiceloss':
            return self._get_beit_dice_loss()
        elif self.loss_type == 'hrnet_ocr':
            return self._get_hrnet_ocr_loss()
        else:
            raise ValueError(f"지원하지 않는 손실함수 타입입니다: {self.loss_type}")

    def _get_bcewithlogitsloss(self):
        return nn.BCEWithLogitsLoss()

    def _get_dice_loss(self):
        return DiceLoss(mode="multilabel")

    def _get_bcewithdice_loss(self):
        d_w = self.loss_params.get('dice_weight', 0.5)
        b_w = self.loss_params.get('bce_weight', 0.5)
        return BCEWithDICE(dice_weight=d_w, bce_weight=b_w)
    
    def _get_hrnet_ocr_loss(self):
        aux_weight = self.loss_params.get('aux_weight', 0.4)
        return HRNetOCRLoss(aux_weight=aux_weight)

    def _get_beit_dice_loss(self):
        smooth = self.loss_params.get('smooth', 1e-6)
        return BeitDiceLoss(smooth=smooth)
    
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
    
class BeitDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BeitDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        if isinstance(predictions, dict):
            main_pred = predictions['out']
            main_loss = self.compute_dice_loss(main_pred, targets)
            return main_loss 
        
        return self.compute_dice_loss(predictions, targets)
    
    def compute_dice_loss(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
class HRNetOCRLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super().__init__()
        self.main_loss = BCEWithDICE(dice_weight=0.7, bce_weight=0.3)
        self.aux_loss = BCEWithDICE(dice_weight=0.7, bce_weight=0.3)
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            main_out = outputs['out']
            aux_out = outputs['aux']
            
            main_loss = self.main_loss(main_out, targets)
            aux_loss = self.aux_loss(aux_out, targets)
            
            total_loss = main_loss + self.aux_weight * aux_loss
            return total_loss
            
        return self.main_loss(outputs, targets)