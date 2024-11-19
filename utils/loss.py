import torch
import torch.nn as nn
from monai.losses import DiceLoss, SoftclDiceLoss, SoftDiceclDiceLoss
from monai.losses import GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss

class LossSelector:
    def __init__(self, loss_config):
        """
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
        elif self.loss_type == 'generalizeddiceloss':
            return self._get_generalized_dice_loss()
        elif self.loss_type == 'softcldiceloss':
            return self._get_softcl_dice_loss()
        elif self.loss_type == 'softdicecldiceloss':
            return self._get_soft_dicecl_dice_loss()
        else:
            raise ValueError(f"지원하지 않는 손실함수 타입입니다: {self.loss_type}")

    def _filter_loss_params(self, required_params):
        """
        필수 파라미터만 남기고 나머지는 제거하는 함수
        :param required_params: 해당 손실 함수에 필요한 파라미터 목록
        :return: 필터링된 파라미터 딕셔너리
        """
        return {key: value for key, value in self.loss_params.items() if key in required_params}

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
        required_params = ['include_background', 'softmax', 'other_act', 'reduction']
        filtered_params = self._filter_loss_params(required_params)
        return DiceLoss(**filtered_params)
    
    def _get_generalized_dice_loss(self):
        """
        GeneralizedDiceLoss
        """
        # GeneralizedDiceLoss에 필요한 파라미터들만 필터링
        required_params = ['include_background', 'softmax', 'other_act', 'w_type', 'reduction']
        filtered_params = self._filter_loss_params(required_params)
        return GeneralizedDiceLoss(**filtered_params)
    
    def _get_softcl_dice_loss(self):
        """
        SoftclDiceLoss: 필요한 파라미터는 iter, smooth, alpha 만
        """
        # SoftclDiceLoss에 필요한 파라미터들만 필터링
        required_params = ['iter', 'smooth']
        filtered_params = self._filter_loss_params(required_params)
        return SoftclDiceLoss(**filtered_params)

    def _get_soft_dicecl_dice_loss(self):
        """
        SoftDiceclDiceLoss: 필요한 파라미터는 iter, smooth, alpha 만
        """
        # SoftDiceclDiceLoss에 필요한 파라미터들만 필터링
        required_params = ['iter', 'smooth', 'alpha']
        filtered_params = self._filter_loss_params(required_params)
        return SoftDiceclDiceLoss(**filtered_params)