import torch.nn as nn
import torchvision.models.segmentation as models 
import segmentation_models_pytorch as smp
import timm
import math
import torch.nn.functional as F

from models.attn_unet import AttU_Net

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

class SMPModel(nn.Module):
    """
    segmentation_models_pytorch에서 제공하는 사전 훈련된 모델을 사용하는 클래스.
    """
    def __init__(
            self,
            model_name: str,
            encoder_name: str,
            num_classes: int,
            encoder_weights: str = 'imagenet'
    ):
        super(SMPModel, self).__init__()
        
        # SMP 모델 초기화
        self.model = smp.__dict__[model_name](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)
    
class TimmModel(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            pretrained: bool = True,
            img_size: int = 384,
            window_size: int = 12
    ):
        super().__init__()
        
        # SwinV2-L 인코더 초기화
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            window_size=window_size,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=True,  
            pretrained_window_sizes=[12, 12, 12, 6]
        )
        
        # 더 큰 채널 수 처리
        self.encoder_channels = self.encoder.feature_info.channels()[-1]  # 1536 for SwinV2-L
        
        # 디코더 수정 (더 큰 채널 수 처리)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder_channels, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, num_classes, 1)
        )

    def forward(self, x):
        # Get features from encoder
        features = self.encoder(x)
        
        # Use the last feature map and adjust channel order
        x = features[-1]
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # Decoder
        x = self.decoder(x)
        
        # Ensure correct output size
        if x.shape[-2:] != (384, 384):
            x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        
        return x.float()  # 출력을 float 타입으로 보장

class ModelSelector:
    def __init__(
        self, 
        config: dict,
        num_classes: int
    ):
        if 'attn_unet' in config['type']:
            self.model = AttU_Net(img_ch=3, output_ch=num_classes)
        
        elif 'torchvision' in config['type']:
            self.model = TorchvisionModel(
                model_name=config["model_name"],
                num_classes=num_classes,
                pretrained=config["pretrained"]
            )
        
        elif 'smp' in config['type']:
            self.model = SMPModel(
                model_name=config["model_name"],
                encoder_name=config["encoder_name"],
                num_classes=num_classes,
                encoder_weights=config["encoder_weights"]
            )
        
        elif 'timm' in config['type']:
            self.model = TimmModel(
                model_name=config["model_name"],
                num_classes=num_classes,
                pretrained=config.get("pretrained", True),
                img_size=config.get("img_size", 384),
                window_size=config.get("window_size", 12)
            )
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:
        return self.model