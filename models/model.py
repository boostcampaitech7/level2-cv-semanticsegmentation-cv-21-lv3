import torch.nn as nn
import torch.cuda
import torchvision.models.segmentation as models 
import segmentation_models_pytorch as smp
import timm
import math
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from transformers import BeitForSemanticSegmentation

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
            model_class = getattr(smp, config["model_name"])

            # 모델 초기화
            self.model = model_class(
                encoder_name=config["encoder_name"],
                encoder_weights=config["encoder_weights"],
                classes=num_classes,
                decoder_attention_type=config.get("decoder_attention_type",None)
            )
        
        elif 'timm' in config['type']:
            self.model = TimmModel(
                model_name=config["model_name"],
                num_classes=num_classes,
                pretrained=config.get("pretrained", True),
                img_size=config.get("img_size", 384),
                window_size=config.get("window_size", 12)
            )
            
        elif 'beit' in config['type']:
            self.model = BeitSegmentationModel(
                model_name=config["model_name"],
                num_classes=num_classes,
                pretrained=config.get("pretrained", True),
                img_size=config.get("img_size", 640),
                original_size=config.get("original_size", 2048),
                output_hidden_states = True
            )
            
        elif 'hrnet_ocr' in config['type']:
            self.model = HRNetOCR(
                num_classes=num_classes,
                pretrained=config.get("pretrained", True)
            )
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:
        return self.model
    
class BeitSegmentationModel(nn.Module):
    def __init__(
             self,
            model_name: str,
            num_classes: int,
            pretrained: bool = True,
            img_size: int = 640,
            original_size: int = 2048,
            output_hidden_states:bool = True,):
        
        super().__init__()
        
        self.model_size = img_size
        self.original_size = original_size
        
        
        # BEiT 모델 초기화
        self.encoder = BeitForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
        # hidden size
        hidden_size = self.encoder.config.hidden_size # 1024
        
        # 디코더 구성
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        


    def forward(self, x):
        batch_size = x.shape[0]
        input_size = x.shape[-2:]
        
        # BEiT forward pass with hidden states
        outputs = self.encoder(x, output_hidden_states=True)
        logits = outputs.logits  # 메인 출력
        
        if self.training and hasattr(outputs, 'hidden_states'):
            # hidden_states shape: [2, 1601, 1024]
            features = outputs.hidden_states[-1]
            # 1601 = 40*40 + 1 (class token)
            seq_len = features.shape[1]
            H = W = int(math.sqrt(seq_len - 1))  # 40x40
            
            # class token 제거하고 reshape
            features = features[:, 1:, :]  # class token 제거
            features = features.reshape(batch_size, H, W, -1)  # [2, 40, 40, 1024]
            features = features.permute(0, 3, 1, 2)  # [2, 1024, 40, 40]
            
            # auxiliary prediction
            aux_logits = self.decoder(features)
            
            # 크기 맞추기
            aux_logits = F.interpolate(aux_logits, size=input_size, mode='bilinear', align_corners=False)
            main_logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            
            return {
                'out': main_logits,
                'aux': aux_logits
            }
        
        # eval 모드 또는 hidden states 없는 경우
        return F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
    
class HRNetOCR(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        pretrained: bool = True,
        in_channels: int = 3
    ):
        super().__init__()
        
        # HRNet from timm
        self.backbone = timm.create_model(
            'hrnet_w48',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # 채널 수 가져오기
        self.channels = self.backbone.feature_info.channels()
        last_inp_channels = sum(self.channels)  # 모든 스테이지의 feature map 합치기
        ocr_mid_channels = 512
        ocr_key_channels = 256
        
        # OCR attention 모듈
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05
        )
        
        # 최종 분류기
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # HRNet features
        features = self.backbone(x)
        
        # 마지막 4개 stage의 feature maps 융합
        out_aux = torch.cat([
            F.interpolate(feature, size=features[-1].shape[2:], mode='bilinear', align_corners=True)
            for feature in features
        ], dim=1)
        
        # Auxiliary head
        aux_out = self.aux_head(out_aux)
        
        # OCR
        feats = self.conv3x3_ocr(out_aux)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(ocr_feats)
        
        # 원본 크기로 복원
        aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)
        
        if self.training:
            return {'out': out, 'aux': aux_out}
        return out
    
class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, channels, height, width = probs.size()
        probs = probs.view(batch_size, self.cls_num, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context

class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.out_channels = out_channels

        self.conv_query = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(key_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, features, context):
        batch_size = features.size(0)

        query = self.conv_query(features).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.conv_key(context).view(batch_size, self.key_channels, -1)
        value = self.conv_value(context).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *features.size()[2:])
        context = self.conv_out(context)

        return context