import random

import wandb
import time
from tqdm import tqdm
import numpy as np
import albumentations as A

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import ModelSelector
from utils.util import save_model, iou, validation, load_config 
from utils.dataset import XRayDataset
from utils.scheduler import LRSchedulerSelector
from utils.loss import LossSelector

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 고정
set_seed(42)

def train(model,
          data_loader,
          val_loader,
          criterion,
          optimizer,
          scheduler,
          num_epochs,
          interver,
          save_dir,
          classes,
          model_name,
          save_name,
          save_group,
          mixed_precision,
          accumulation_steps,
          config
    ):

    print(f'Start training..')
    best_dice = 0.
    best_iou = 0.
    scaler = torch.cuda.amp.GradScaler()

    # WandB 초기화
    wandb.init(
        entity="ppp6131-yonsei-university",
        project="boostcamp7th_semantic_segmentation",
        config=config,
        name=model_name,
        group=save_group
    )
    
    # 모델 구조 로깅
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_iou_loss = 0.0
        
        # Training loop
        for step, (images, masks) in tqdm(enumerate(data_loader), 
                                        total=len(data_loader), 
                                        desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            
            # Loss 계산
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()

            # IoU Loss 계산
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float() # threshold 적용
            iou_loss_value = (1.0 - iou(masks, outputs)).mean()
            epoch_iou_loss += iou_loss_value.item()
            
            # Backpropagation
            if mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient Accumulation
            if(step + 1) % accumulation_steps == 0  or (step + 1) == len(data_loader):
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
        
        # 에폭당 평균 train loss / train iou loss 계산
        avg_train_loss = epoch_loss / len(data_loader)
        avg_train_iou_loss = epoch_iou_loss / len(data_loader)

        # 로그 출력
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Train Loss: {avg_train_loss:.4f}, Avg IoU Loss: {avg_train_iou_loss:.4f}")
            
        # Validation 단계
        if (epoch + 1) % interver == 0:
            # utils.validation에서 구현된 validation 함수 사용
            dice, IoU = validation(epoch + 1, model, val_loader, criterion, classes)
            
            # WandB에 메트릭 로깅
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train_loss": avg_train_loss,
                "train_iou_loss": avg_train_iou_loss,
                "dice_coefficient": dice,
                "iou": IoU
            })
            
            # Save CheckPoints
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                best_dice = dice
                best_iou = IoU

                # model unhooking
                wandb.unwatch(model)
                save_model(model, save_dir, model_name)
                
                # Best 모델을 WandB에 저장 -> Test 모델 형태랑 충돌
                artifact = wandb.Artifact(
                    name=f"{save_name}_best", 
                    type="model",
                    description=f"Best model with dice score: {dice:.4f}, iou score: {IoU:.4f}"
                )
                artifact.add_file(f"{save_dir}/{model_name}_best_model.pt")
                wandb.log_artifact(artifact)

    print(f"Training completed. Best Dice Score: {best_dice:.4f}, Best IoU Score: {best_iou:.4f}")

def main():
    # WandB 로그인
    wandb.login()
    
    tf = A.Resize(512, 512)
    
    config = load_config('./config/config.json')  # config.json 로드
    data_root = config['DATA_ROOT']
    img_root = f"{data_root}/train/DCM"
    label_root = f"{data_root}/train/outputs_json"
    classes = config['CLASSES']
    batch_size = config['BATCH_SIZE']
    n_class = len(classes)
    model_name = config['model']['model_name']
    save_name = f'{model_name}_{config["LR"]}_{batch_size}_'+time.strftime("%Y%m%d_%H%M%S")
        # if not 'lr_scheduler' in config else \
        #     f'{model_name}_{config["lr_scheduler"]['type']}_{batch_size}_'+time.strftime("%Y%m%d_%H%M%S")
    save_group = model_name # model name, augmentation, scheduler

    train_dataset = XRayDataset(
        is_train=True,
        transforms=tf,
        img_root=img_root,
        label_root=label_root,
        classes=classes
    )
    valid_dataset = XRayDataset(
        is_train=True,
        transforms=tf,
        img_root=img_root,
        label_root=label_root,
        classes=classes
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    # 주의: validation data는 이미지 크기가 크기 때문에 `num_workers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # 모델 정의
    model_selector = ModelSelector(
        config=config['model'],
        num_classes=n_class
    )
    model = model_selector.get_model()
    model = model.cuda()
    
    # Scheduler 설정
    scheduler = None
    # if 'lr_scheduler' in config:
    #     scheduler_selector = LRSchedulerSelector(optimizer, config['lr_scheduler'])
    #     scheduler = scheduler_selector.get_scheduler()
    
    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()
    # loss_selector = LossSelector(config['loss'])
    # criterion = loss_selector.get_loss()

    # Optimizer를 정의합니다.
    optimizer = optim.Adam(params=model.parameters(), lr=config['LR'], weight_decay=1e-6)

    # mixed precision 설정
    if 'mixed_precision' in config:
        mixed_precision = config['mixed_precision']
    else:
        mixed_precision = True

    # Gradient Accumulation_steps 설정
    if 'accumulation_steps' in config:
        accumulation_steps = config['accumulation_steps']
    else:
        accumulation_steps = 1

    # wandb에 기록할 config 설정
    wandb_config = {
        "learning_rate": config['LR'],
        "batch_size": batch_size,
        "optimizer": "Adam",
        "weight_decay": 1e-6,
        "model": model_name,
        "epochs": config['NUM_EPOCHS'],
        "image_size": 512,
        "validation_interval": config['VAL_INTERVER']
    }

    train(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=config['NUM_EPOCHS'],
        interver=config['VAL_INTERVER'],
        save_dir=config['SAVED_DIR'],
        classes=config['CLASSES'],
        model_name=model_name,
        save_name=save_name,
        save_group=save_group,
        config=wandb_config,
        mixed_precision=mixed_precision,
        accumulation_steps=accumulation_steps
    )

if __name__ == '__main__':
    main()