import argparse

import wandb

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import ModelSelector
from utils.util import load_config 
from utils.dataset import XRayDataset
from utils.loss import LossSelector
from utils.transforms import get_transforms
from trainer.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model file')
    args = parser.parse_args()
    model_path = args.model_path   
    # WandB 로그인
    wandb.login()
    
    train_tf = get_transforms(is_train=True)
    valid_tf = get_transforms(is_train=False)

    data_root = config['DATA_ROOT']
    img_root = f"{data_root}/train/DCM"
    label_root = f"{data_root}/train/outputs_json"
    classes = config['CLASSES']
    batch_size = config['BATCH_SIZE']
   
    train_dataset = XRayDataset(
        is_train=True,
        transforms=train_tf,
        img_root=img_root,
        label_root=label_root,
        classes=classes
    )
    valid_dataset = XRayDataset(
        is_train=False,
        transforms=valid_tf,
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
        pin_memory=True,
        drop_last=False
    )

    if model_path is None:
        # model
        model_selector = ModelSelector(
            config=config['model'],
            num_classes=len(classes)
        )
        model = model_selector.get_model()
    else: # fine tuning
        model = torch.load(model_path)
    model = model.cuda()
    
    # Loss function
    #criterion = DiceLoss(mode="multilabel")
    loss_selector = LossSelector(config['loss'])
    criterion = loss_selector.get_loss()

    # Optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=config['LR'], weight_decay=1e-6)
    
    # Trainer 클래스 인스턴스 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        sweep_mode=sweep_mode  # sweep_mode 전달
    )
    
    trainer.train()

if __name__ == '__main__':
    
    # config.json 로드
    config = load_config('./config/config.json')  
    sweep_mode = config.get('sweep_config', {}).get('sweep_mode', False)  # sweep_mode 값 가져오기
    print(f'Sweep_mode: {sweep_mode}')

    # run sweep
    if sweep_mode:
        sweep_id = wandb.sweep(config['sweep_config'],
                               project="boostcamp7th_semantic_segmentation", 
                               entity="ppp6131-yonsei-university")
        wandb.agent(sweep_id, function=lambda:main())  # 각 모델에 대해 한 번씩 실행
    else:
        # 학습 시작
        main()

    