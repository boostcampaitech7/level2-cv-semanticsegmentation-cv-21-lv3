

import wandb
from tqdm import tqdm
import albumentations as A

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import ModelSelector
from utils.util import load_config 
from utils.dataset import XRayDataset
from utils.scheduler import LRSchedulerSelector
from utils.loss import LossSelector
from trainer.trainer import Trainer  # Trainer 클래스 임포트

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
   
    train_dataset = XRayDataset(
        is_train=True,
        transforms=tf,
        img_root=img_root,
        label_root=label_root,
        classes=classes
    )
    valid_dataset = XRayDataset(
        is_train=False,
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

    # model
    model_selector = ModelSelector(
        config=config['model'],
        num_classes=len(classes)
    )
    model = model_selector.get_model()
    model = model.cuda()
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    # loss_selector = LossSelector(config['loss'])
    # criterion = loss_selector.get_loss()

    # Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=config['LR'], weight_decay=1e-6)

    # Trainer 클래스 인스턴스 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config
    )

    # 학습 시작
    trainer.train()

if __name__ == '__main__':
    main()