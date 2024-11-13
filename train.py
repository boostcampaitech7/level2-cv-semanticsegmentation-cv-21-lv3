import random

import tqdm
import numpy as np
import albumentations as A

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import ModelSelector
from utils.util import  save_model, validation, load_config 
from utils.dataset import XRayDataset

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
          model_name
    ):

    print(f'Start training..')
    best_dice = 0.
    
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            
            # Loss
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #Valid Interver     
        if (epoch + 1) % interver == 0:
            dice = validation(epoch + 1, model, val_loader, criterion, classes)
            # Save CheckPoints
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                best_dice = dice
                save_model(model,save_dir,model_name)

def main():
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

    train(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        num_epochs = config['NUM_EPOCHS'],
        interver = config['VAL_INTERVER'],
        save_dir = config['SAVED_DIR'],
        classes = config['CLASSES'],
        model_name = config['model']['model_name']
    )

if __name__ == '__main__':
    main()