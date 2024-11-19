import os
import time

import wandb
from tqdm import tqdm

import torch

from utils.scheduler import LRSchedulerSelector
from utils.util import iou, validation, set_seed

class Trainer:
    """
    Trainer Class
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config):
        """
        Initialize the Trainer class.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_epochs = config['NUM_EPOCHS']
        self.interver = config['VAL_INTERVER']
        self.best_dice = 0.
        self.best_iou = 0.
        self.model_name = config['model']['model_name']
        self.save_dir = config['SAVED_DIR']
        self.classes = config['CLASSES']

        # 저장될 모델 이름 설정
        self.save_name = f'{self.model_name}_{self.config["LR"]}_{self.config["BATCH_SIZE"]}_'+time.strftime("%Y%m%d_%H%M%S") \
        if not 'lr_scheduler' in self.config else \
        f'{self.model_name}_{self.config["lr_scheduler"]["type"]}_{self.config["BATCH_SIZE"]}_'+time.strftime("%Y%m%d_%H%M%S")

        # mixed precision 설정
        if 'mixed_precision' in config:
            self.mixed_precision = config['mixed_precision']
        else:
            self.mixed_precision = True

        # Gradient Accumulation_steps 설정
        if 'accumulation_steps' in config:
            self.accumulation_steps = config['accumulation_steps']
        else:
            self.accumulation_steps = 1

        # Scheduler 설정
        if 'lr_scheduler' in config:
            scheduler_selector = LRSchedulerSelector(optimizer, config['lr_scheduler'])
            self.scheduler = scheduler_selector.get_scheduler()
        else:
            self.scheduler = None

    def _train_epoch(self, epoch):
        """
        Train the model for one epoch.

        :param epoch: The current epoch number.
        :return: avg_train_loss: float, the average training loss
                avg_train_iou_loss: float, the average IoU loss
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_iou_loss = 0.0
        
        # Training loop
        for step, (images, masks) in tqdm(enumerate(self.train_loader), 
                                        total=len(self.train_loader), 
                                        desc=f'Epoch {epoch+1}/{self.num_epochs}'):
            images, masks = images.cuda(), masks.cuda()
            outputs = self.model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            
            # Loss 계산
            loss = self.criterion(outputs, masks)
            epoch_loss += loss.item()

            # IoU Loss 계산
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float() # threshold 적용
            iou_loss_value = (1.0 - iou(masks, outputs)).mean()
            epoch_iou_loss += iou_loss_value.item()
            
            self._backpropagate(loss, step)
        
        # 에폭당 평균 train loss / train iou loss 계산
        avg_loss = epoch_loss / len(self.train_loader)
        avg_iou_loss = epoch_iou_loss / len(self.train_loader)

        # 로그 출력
        print(f"Epoch [{epoch+1}/{epoch}] - Avg Train Loss: {avg_loss:.4f}, Avg IoU Loss: {avg_iou_loss:.4f}")
            
        return avg_loss, avg_iou_loss

    def _backpropagate(self, loss, step):
        """
        Backpropagate the loss.

        :param loss: The calculated loss value
        :param step: The current step number
        """
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
            self._optimizer_step()

    def _optimizer_step(self):
        """
        Perform an optimizer step.
        """
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def valid(self, epoch, avg_train_loss, avg_train_iou_loss):
        """
        Perform validation and save the best model.

        :param epoch: The current epoch number
        :param avg_train_loss: The average training loss
        :param avg_train_iou_loss: The average IoU loss
        """
        # utils.validation에서 구현된 validation 함수 사용
        dice, IoU = validation(epoch + 1, self.model, self.val_loader, self.criterion, self.classes)
        
        # WandB에 메트릭 로깅
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "train_loss": avg_train_loss,
            "train_iou_loss": avg_train_iou_loss,
            "dice_coefficient": dice,
            "iou": IoU
        })
        
        # Save CheckPoints
        if self.best_dice < dice:
            self._save_best_model(epoch, dice, IoU)

    def _save_best_model(self, epoch, dice, IoU):
        """
        Save the best model.

        :param epoch: The current epoch number
        :param dice: The Dice score of the current epoch
        :param IoU: The IoU score of the current epoch
        """
        print(f"Best performance at epoch: {epoch + 1}, {self.best_dice:.4f} -> {dice:.4f}")
        self.best_dice = dice
        self.best_iou = IoU
        
        # 모델 저장
        self.save_checkpoint(dice, IoU)

    def save_checkpoint(self, dice, IoU):
        """
        Save the model checkpoint.

        :param dice: The Dice score of the current epoch
        :param IoU: The IoU score of the current epoch
        """
        
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 모델 저장
        wandb.unwatch(self.model)
        output_path = os.path.join(self.save_dir, f'{self.save_name}.pt')
        torch.save(self.model, output_path)
        print(f"Model saved to {output_path}")

        # WandB에 메트릭 로깅
        artifact = wandb.Artifact(
            name=self.save_name, 
            type="model",
            description=f"Best model with dice score: {dice:.4f}, iou score: {IoU:.4f}"
        )
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)

    def train(self):
        """
        Perform the entire training process.
        """

        # 시드 고정
        set_seed(42)

        # WandB 초기화
        wandb.init(
            entity="ppp6131-yonsei-university",
            project="boostcamp7th_semantic_segmentation",
            config=self.config,
            name=self.save_name,
            group=self.model_name
        )

        # 학습 수행
        for epoch in range(self.num_epochs):
            # 모델 구조 로깅
            wandb.watch(self.model, self.criterion, log="all", log_freq=100)
            avg_train_loss, avg_train_iou_loss = self._train_epoch(epoch)
            if (epoch + 1) % self.interver == 0:
                self.valid(epoch, avg_train_loss, avg_train_iou_loss)
        print(f"Training completed. Best Dice Score: {self.best_dice:.4f}, Best IoU Score: {self.best_iou:.4f}")