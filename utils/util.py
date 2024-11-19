import os
import json

import tqdm
import numpy as np

import torch
import torch.nn.functional as F
# 설정 파일 로드
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def save_model(model, save_root, model_name=None):
    # 모델 이름을 지정하지 않으면 클래스 이름으로 설정
    if model_name is None:
        model_name = model.__class__.__name__

    # 파일명을 모델 이름에 맞게 설정
    file_name = f"{model_name}_best_model.pt"
    
    # 저장 디렉토리 생성
    os.makedirs(save_root, exist_ok=True)
    
    # 모델 저장
    output_path = os.path.join(save_root, file_name)
    torch.save(model, output_path)
    print(f"Model saved to {output_path}")

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def iou(y_true, y_pred): 
    """
    IoU (Intersection over Union) 계산 함수
    :param y_true: Ground truth tensor
    :param y_pred: Predicted tensor
    :return: IoU 값 (Tensor 형태)
    """
    # Flatten tensors along the spatial dimensions
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    
    # Calculate intersection
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    # Calculate union
    union = torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) - intersection
    
    # Small epsilon to avoid division by zero
    eps = 0.0001
    
    # Calculate IoU
    return (intersection + eps) / (union + eps)

def validation(epoch, model, data_loader, criterion, CLASSES, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    ious = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0


        for step, (images, masks) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()

            outputs = model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            # Loss 계산
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            # threshold 적용
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach()
            masks = masks.detach()

            # Dice Coefficient 계산
            dice = dice_coef(outputs, masks)
            dices.append(dice)

            # IoU 계산
            iou_value = iou(masks, outputs)
            ious.append(iou_value)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    ious = torch.cat(ious, 0)
    ious_per_class = torch.mean(ious, 0)
    metrics_str = [
        f"{c:<12}: Dice={d.item():.4f}, IoU={i.item():.4f}"
        for c, d, i in zip(CLASSES, dices_per_class, ious_per_class)
    ]
    metrics_str = "\n".join(metrics_str)
    print(metrics_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    avg_iou = torch.mean(ious_per_class).item()

    return avg_dice, avg_iou