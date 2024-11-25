import wandb
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import ModelSelector
from utils.util import load_config 
from utils.dataset import XRayDataset
from utils.scheduler import LRSchedulerSelector
from utils.loss import LossSelector
from trainer.trainer import Trainer  # Trainer 클래스 임포트
import argparse

from lion_pytorch import Lion

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model configuration from command line.')
    parser.add_argument('--model_name', type=str, help='Name of the model', default='FPN')
    parser.add_argument('--encoder_name', type=str, help='Name of the encoder', default='resnet152')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use', default='adam')
    parser.add_argument('--resize', type=int, help='Reshape size', default=512)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    return parser.parse_args()

def main():
    args = parse_arguments()

    # WandB 로그인
    wandb.login()
    
    resize = args.resize
    # 데이터 증강 및 전처리
    tf = A.Resize(resize, resize)
    print('##########################')
    print(f'resize: {resize}')
    print('##########################')

    config = load_config('./config/config.json')  # config.json 로드
    data_root = config['DATA_ROOT']
    img_root = f"{data_root}/train/DCM"
    label_root = f"{data_root}/train/outputs_json"
    classes = config['CLASSES']
    batch_size = args.batch_size
   
    config['model']['model_name'] = args.model_name if args.model_name else config['model']['model_name']
    config['model']['encoder_name'] = args.encoder_name if args.encoder_name else config['model']['encoder_name']

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
        num_workers=3,
        drop_last=True,
    )
    # 주의: validation data는 이미지 크기가 크기 때문에 `num_workers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
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
    # criterion = nn.BCEWithLogitsLoss()
    from segmentation_models_pytorch.losses import *
    # loss_selector = LossSelector(config['loss'])
    criterion = DiceLoss(mode="multilabel")
    print(f'criterion is: {criterion}')
    # Optimizer
    if args.optimizer.lower() == 'lion':
        optimizer = Lion(params=model.parameters(), lr=config['LR'], weight_decay=1e-6)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(), lr=config['LR'], weight_decay=1e-6)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=config['LR'], momentum=0.9, weight_decay=1e-6)
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=config['LR'], weight_decay=1e-6)

    print(f'optimizer: {optimizer}, model_name: {config["model"]["model_name"]}, encoder_name: {config["model"]["encoder_name"]}')

    # Trainer 클래스 인스턴스 생성
    sweep_mode = config.get('sweep_config', {}).get('sweep_mode', False)  # sweep_mode 값 가져오기
    print(f'sweep_mode: {sweep_mode}')
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        sweep_mode=sweep_mode  # sweep_mode 전달
    )

    # run sweep
    if sweep_mode:
        sweep_id = wandb.sweep(config['sweep_config'],
                               project="boostcamp7th_semantic_segmentation", 
                               entity="ppp6131-yonsei-university")
        wandb.agent(sweep_id, function=trainer.train, count=1)  # 각 모델에 대해 한 번씩 실행
    else:
        # 학습 시작
        trainer.train()

if __name__ == '__main__':
    main()