import argparse

import wandb
from tqdm import tqdm
import albumentations as A
from segmentation_models_pytorch.losses import DiceLoss

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import ModelSelector
from utils.util import load_config 
from utils.dataset import XRayDataset, SwinXRayDataset, BeitXRayDataset, HRNetXRayDataset
from utils.scheduler import LRSchedulerSelector
from utils.loss import LossSelector
from utils.transforms import get_transforms
from trainer.trainer import Trainer, SwinTrainer, BeitTrainer, HRNetTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model file')
    args = parser.parse_args()
    model_path = args.model_path   
    wandb.login()
    
    # Transform 설정
    train_tf = get_transforms(config=config,is_train=True)
    valid_tf = get_transforms(config=config,is_train=False)

    # 데이터 경로 설정
    data_root = config['DATA_ROOT']
    img_root = f"{data_root}/train/DCM"
    label_root = f"{data_root}/train/outputs_json"
    classes = config['CLASSES']
    batch_size = config['BATCH_SIZE']
    
    # Dataset 선택 (model type에 따라)
    if config['model']['type'] == 'beit' :
        Dataset = BeitXRayDataset
    elif config['model']['type'] == 'HRnet':
        Dataset = HRNetXRayDataset
    else :
        Dataset = XRayDataset
    
    train_dataset = Dataset(
        is_train=True,
        transforms=train_tf,
        img_root=img_root,
        label_root=label_root,
        classes=classes,
        config=config  # Swin config 전달
    )
    valid_dataset = Dataset(
        is_train=False,
        transforms=valid_tf,
        img_root=img_root,
        label_root=label_root,
        classes=classes,
        config=config  # Swin config 전달
    )

    train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=config['BATCH_SIZE'],
            shuffle=True,
            num_workers=4,  # 메모리 사용량 고려하여 조정
            pin_memory=True,
            persistent_workers=True,  # worker 재사용
            prefetch_factor=2
        )
    valid_loader = DataLoader(
            dataset=valid_dataset, 
            batch_size=config.get('VAL_BATCH_SIZE', 2),  # validation 배치 사이즈 별도 설정
            shuffle=False,
            num_workers=4,  # 메모리 사용량 고려하여 조정
            pin_memory=True,
            drop_last=False
        )

    # 모델 초기화
    if model_path:
        model = torch.load(model_path)
    else:
        model_selector = ModelSelector(
            config=config['model'],
            num_classes=len(classes)
        )
        model = model_selector.get_model()
    model = model.cuda()
    
    # Loss function
    loss_selector = LossSelector(config['loss'])
    criterion = loss_selector.get_loss()

    # Optimizer - config에서 설정 가져오기
    optimizer_config = config.get('optimizer', {})
    optimizer = optim.AdamW(
        params=model.parameters(), 
        lr=config['LR'],
        weight_decay=optimizer_config.get('weight_decay', 0.05)  # Swin 기본값
    )
    
    # Trainer 선택 (model type에 따라)
    # Dataset 선택 (model type에 따라)
    if config['model']['type'] == 'beit' :
        TrainerClass = BeitTrainer
    elif config['model']['type'] == 'HRnet':
        TrainerClass = HRNetTrainer
    else :
        TrainerClass = Trainer
    
    trainer = TrainerClass(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        sweep_mode=sweep_mode
    )
    
    trainer.train()

if __name__ == '__main__':
    config = load_config('/data/ephemeral/home/ahn/config/config.json')  
    sweep_mode = config.get('sweep_config', {}).get('sweep_mode', False)
    print(f'Sweep_mode: {sweep_mode}')

    if sweep_mode:
        sweep_id = wandb.sweep(config['sweep_config'],
                                project="boostcamp7th_semantic_segmentation", 
                                entity="ppp6131-yonsei-university")
        wandb.agent(sweep_id, function=lambda:main())
    else:
        main()