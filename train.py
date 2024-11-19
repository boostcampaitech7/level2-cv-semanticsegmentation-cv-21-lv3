import random
import time

import wandb
from tqdm import tqdm
import numpy as np
import albumentations as A

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import ModelSelector
from utils.util import save_model, validation, load_config 
from utils.dataset import XRayDataset
from utils.loss import LRSchedulerSelector

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
        
        # Training loop
        for step, (images, masks) in tqdm(enumerate(data_loader), 
                                        total=len(data_loader), 
                                        desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, masks = images.cuda(), masks.cuda()

            # Train loss
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(images)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']
                loss = criterion(outputs, masks)
                epoch_loss += loss.item()

            # Backpropagation
            if mixed_precision:   
                scaler.scale(loss).backward()               
            else:
                loss.backward()
            
            # Gradient Accumulation
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        # 에폭당 평균 train loss 계산
        avg_train_loss = epoch_loss / len(data_loader)
        print(f"Epoch[{epoch+1}/{num_epochs}] Completed. Avg Loss: {avg_train_loss:.4f}")
        
        # Validation 단계
        if (epoch + 1) % interver == 0:
            dice = validation(epoch + 1, model, val_loader, criterion, classes)
            
            # WandB에 메트릭 로깅
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train_loss": avg_train_loss,
                "dice_coefficient": dice
            })
            
            # Save CheckPoints
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                best_dice = dice
                
                # model unhooking
                wandb.unwatch(model)
                save_model(model, save_dir, save_name)
                
                # Best 모델을 WandB에 저장
                artifact = wandb.Artifact(
                    name=f"{save_name}_best", 
                    type="model",
                    description=f"Best model with dice score: {dice:.4f}"
                )
                artifact.add_file(f"{save_dir}/{save_name}_best_model.pt")
                wandb.log_artifact(artifact)

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
    save_name = f'{model_name}_{config["LR"]}_{batch_size}'+time.strftime("%Y%m%d_%H%M%S") if not config['lr_scheduler'] else f'{model_name}_{config["lr_scheduler"]}_{batch_size}'+time.strftime("%Y%m%d_%H%M%S")
    save_group = model_name

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
    
    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer를 정의합니다.
    optimizer = optim.Adam(params=model.parameters(), lr=config['LR'], weight_decay=1e-6)
    
    # mixed_precision 설정
    mixed_precision = config.get('mixed_precision', True)

    # Gradient Accumulation_steps 설정
    accumulation_steps = config.get('accumulation_steps', 1)

    # wandb에 기록할 config 설정
    wandb_config = {
        "learning_rate": config['LR'],
        "batch_size": batch_size,
        "optimizer": "Adam",
        "weight_decay": 1e-6,
        "model": config['save']['name'],
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