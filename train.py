import argparse

import wandb
import torch
from torch.utils.data import DataLoader

from models.model import ModelSelector
from utils.util import load_config 
<<<<<<< HEAD
from utils.dataset import XRayDataset
from utils.loss import LossSelector
from utils.transforms import get_transforms
from trainer.trainer import Trainer
from utils.optimizer import get_optimizer
=======
from utils.dataset import XRayDataset, SwinXRayDataset, BeitXRayDataset, HRNetXRayDataset
from utils.scheduler import LRSchedulerSelector
from utils.loss import LossSelector
from utils.transforms import get_transforms
from trainer.trainer import Trainer, SwinTrainer, BeitTrainer, HRNetTrainer
>>>>>>> Ahn-latte/issue16



def parse_arguments():
    parser = argparse.ArgumentParser(description='Model configuration from command line.')
    parser.add_argument('--model_name', type=str, help='Name of the model', default='FPN')
    parser.add_argument('--encoder_name', type=str, help='Name of the encoder', default='resnet152')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use', default='adam')
    parser.add_argument('--resize', type=int, help='Reshape size', default=512)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model file')
<<<<<<< HEAD

    return parser.parse_args()

def main():
    args = parse_arguments()
    # WandB 로그인
    wandb.login()
    
    train_tf = get_transforms(is_train=True, size=args.resize)
    valid_tf = get_transforms(is_train=False, size=args.resize)
=======
    args = parser.parse_args()
    model_path = args.model_path   
    wandb.login()
    
    # Transform 설정
    train_tf = get_transforms(config=config,is_train=True)
    valid_tf = get_transforms(config=config,is_train=False)
>>>>>>> Ahn-latte/issue16

    # 데이터 경로 설정
    data_root = config['DATA_ROOT']
    img_root = f"{data_root}/train/DCM"
    label_root = f"{data_root}/train/outputs_json"
    classes = config['CLASSES']
<<<<<<< HEAD
    batch_size = args.batch_size
    
    config['model']['model_name'] = args.model_name if args.model_name else config['model']['model_name']
    config['model']['encoder_name'] = args.encoder_name if args.encoder_name else config['model']['encoder_name']
   
    train_dataset = XRayDataset(
=======
    batch_size = config['BATCH_SIZE']
    
    # Dataset 선택 (model type에 따라)
    if config['model']['type'] == 'beit' :
        Dataset = BeitXRayDataset
    elif config['model']['type'] == 'HRnet':
        Dataset = HRNetXRayDataset
    else :
        Dataset = XRayDataset
    
    train_dataset = Dataset(
>>>>>>> Ahn-latte/issue16
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

<<<<<<< HEAD
    if args.model_path is None:
        # model
=======
    # 모델 초기화
    if model_path:
        model = torch.load(model_path)
    else:
>>>>>>> Ahn-latte/issue16
        model_selector = ModelSelector(
            config=config['model'],
            num_classes=len(classes)
        )
        model = model_selector.get_model()
<<<<<<< HEAD
    else: # fine tuning
        model = torch.load(args.model_path)
=======
>>>>>>> Ahn-latte/issue16
    model = model.cuda()
    
    # Loss function
    loss_selector = LossSelector(config['loss'])
    criterion = loss_selector.get_loss()

<<<<<<< HEAD
    # Use the get_optimizer function from optimizer.py
    optimizer = get_optimizer(args.optimizer, model.parameters(), config['LR'])
    print(f'optimizer: {optimizer}, model_name: {config["model"]["model_name"]}, encoder_name: {config["model"]["encoder_name"]}')
=======
    # Optimizer - config에서 설정 가져오기
    optimizer_config = config.get('optimizer', {})
    optimizer = optim.AdamW(
        params=model.parameters(), 
        lr=config['LR'],
        weight_decay=optimizer_config.get('weight_decay', 0.05)  # Swin 기본값
    )
>>>>>>> Ahn-latte/issue16
    
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