import random
import wandb
import tqdm
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
          scheduler,
          num_epochs,
          interver,
          save_dir,
          classes,
          model_name,
          config
    ):
    
    # WandB 초기화
    wandb.init(
        project="xray-segmentation",
        config=config,
        name=model_name,
    )
    
    # 모델 구조 로깅
    wandb.watch(model, criterion, log="all", log_freq=100)

    print(f'Start training..')
    best_dice = 0.
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training loop
        for step, (images, masks) in tqdm.tqdm(enumerate(data_loader), 
                                             total=len(data_loader), 
                                             desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            
            # Loss 계산
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 에폭당 평균 train loss 계산
        avg_train_loss = epoch_loss / len(data_loader)
            
        # Validation 단계
        if (epoch + 1) % interver == 0:
            # utils.validation에서 구현된 validation 함수 사용
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
                save_model(model, save_dir, model_name)
                
                # Best 모델을 WandB에 저장
                artifact = wandb.Artifact(
                    name=f"{model_name}_best", 
                    type="model",
                    description=f"Best model with dice score: {dice:.4f}"
                )
                artifact.add_file(f"{save_dir}/{model_name}.pth")
                wandb.log_artifact(artifact)

def main():
    # WandB 로그인
    wandb.login()
    
    tf = A.Resize(512, 512)
    
    config = load_config('./config/config.json')  # config.json 로드
    img_root = config['IMAGE_ROOT']
    label_root = config['LABEL_ROOT']
    classes = config['CLASSES']
    batch_size = config['BATCH_SIZE']
    n_class = len(classes)

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
    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=0,
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

    # wandb에 기록할 config 설정
    wandb_config = {
        "learning_rate": config['LR'],
        "batch_size": batch_size,
        "optimizer": "Adam",
        "weight_decay": 1e-6,
        "model": config['model']['model_name'],
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
        model_name=config['model']['model_name'],
        config=wandb_config
    )

if __name__ == '__main__':
    main()