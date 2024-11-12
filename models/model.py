import torch.nn as nn
import torchvision.models.segmentation as models 

class TorchvisionModel(nn.Module):
    """
    Torchvision에서 제공하는 사전 훈련된 모델을 사용하는 클래스.
    """
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            pretrained: bool
    ):
        super(TorchvisionModel, self).__init__()
        self.model = models.__dict__[model_name](pretrained=pretrained)
        
        # 모델의 최종 분류기 부분을 사용자 정의 클래스 수에 맞게 조정
        if hasattr(self.model, 'classifier'):
            # FCN이나 DeepLab 같은 segmentation 모델에서 classifier 부분 수정
            num_ftrs = self.model.classifier[-1].in_channels
            self.model.classifier[-1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1))

        if hasattr(self.model, 'aux_classifier'):
            # 보조 분류기(aux_classifier)도 클래스 수에 맞게 수정
            num_aux_ftrs = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(num_aux_ftrs, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        return self.model(x)

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):
        
        # 모델 유형에 따라 적절한 모델 객체를 생성
        if model_type == 'simple':
            self.model = None
        
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'timm':
            self.model = None
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model