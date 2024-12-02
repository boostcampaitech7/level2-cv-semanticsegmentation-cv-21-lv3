import os
import json
import random

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

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 1e-7
    
    dice = (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)
    return dice

def iou(y_true, y_pred): 
    """
    IoU (Intersection over Union) 계산 함수
    :param y_true: Ground truth tensor
    :param y_pred: Predicted tensor
    :return: IoU 값 (Tensor 형태)
    """
    y_true_f = y_true.flatten(2).float()
    y_pred_f = y_pred.flatten(2).float()
    
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    union = torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) - intersection
    
    eps = 1e-7  # epsilon 값을 더 작게 조정
    return (intersection + eps) / (union + eps)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def validation_swin(epoch, model, data_loader, criterion, CLASSES, config, thr=0.5):
    """
    Swin Transformer를 위한 validation 함수
    여러 threshold 값에 대해 검증
    """
    print(f'Start validation #{epoch:2d}')
    model.eval()

    # Swin config에서 interpolation 모드 가져오기
    interpolation_mode = config.get('interpolation_mode', 'bilinear')
    
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
            
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, 
                                     size=(mask_h, mask_w), 
                                     mode=interpolation_mode)
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.75).detach()
            masks = masks.detach()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

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