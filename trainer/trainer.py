import os
import time

import wandb
from tqdm import tqdm

import torch

from utils.scheduler import LRSchedulerSelector
from utils.util import iou, validation, set_seed, validation_swin

class Trainer:
    """
    Trainer Class
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config, sweep_mode):
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
        self.sweep_mode = sweep_mode
        
        # 하이퍼파라미터 초기화
        self.num_epochs = config['NUM_EPOCHS']
        self.interver = config['VAL_INTERVER']
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.model_name = config['model']['model_name']
        self.save_dir = config['SAVED_DIR']
        self.classes = config['CLASSES']
        self.lr = self.config["LR"]
        self.batch_size = self.config["BATCH_SIZE"]

        # mixed precision 및 Gradient Accumulation_steps 설정
        self.mixed_precision = config.get('mixed_precision', True)
        self.accumulation_steps = config.get('accumulation_steps', 1)

        # Scheduler 설정
        self.scheduler = LRSchedulerSelector(optimizer, config.get('lr_scheduler')).get_scheduler() if 'lr_scheduler' in config else None

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
            outputs = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
            
            # Loss 계산
            loss = self.criterion(outputs, masks)
            epoch_loss += loss.item()

            # IoU Loss 계산
            outputs = (torch.sigmoid(outputs) > 0.5).float()  # threshold 적용
            epoch_iou_loss += (1.0 - iou(masks, outputs)).mean().item()
            
            self._backpropagate(loss, step)
        
        # 에폭당 평균 train loss / train iou loss 계산
        avg_loss = epoch_loss / len(self.train_loader)
        avg_iou_loss = epoch_iou_loss / len(self.train_loader)

        # 로그 출력
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_loss,
            "avg_iou_loss": avg_iou_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        })
        print(f"Epoch [{epoch+1}/{self.num_epochs}] - Avg Train Loss: {avg_loss:.4f}, Avg IoU Loss: {avg_iou_loss:.4f}")
            
        return avg_loss, avg_iou_loss

    def _backpropagate(self, loss, step):
        """
        Backpropagate the loss.

        :param loss: The calculated loss value
        :param step: The current step number
        """
        self.scaler.scale(loss).backward() if self.mixed_precision else loss.backward()

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
        dice, IoU = validation(epoch + 1, self.model, self.val_loader, self.criterion, self.classes)
        
        # WandB에 메트릭 로깅
        # WandB에 메트릭 로깅
        wandb.log({
            "epoch": epoch + 1,
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
        os.makedirs(self.save_dir, exist_ok=True)       

        # 모델 저장
        wandb.unwatch(self.model)
        output_path = os.path.join(self.save_dir, f'{self.save_name}.pt')
        torch.save(self.model, output_path)
        print(f"Model saved to {output_path}")


    def train(self):
        """
        Perform the entire training process.
        
        :param sweep_mode: Boolean indicating if sweep mode is active.
        """

        # 시드 고정
        set_seed(42)

        # sweep 하이퍼파라미터 설정
        if self.sweep_mode:
            self.num_epochs = wandb.config.num_epochs
            self.batch_size = wandb.config.batch_size
            self.optimizer.param_groups[0]['lr'] = wandb.config.learning_rate
            print(f"lr:{self.optimizer.param_groups[0]['lr']}, batch_size:{self.batch_size}")

        # 저장될 모델 이름 설정
        self.lr_scheduler_type = self.config.get("lr_scheduler", {}).get("type", "default") 
        self.save_name = f'{self.model_name}_{self.lr}_{self.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}' \
                         if not 'lr_scheduler' in self.config else \
                         f'{self.model_name}_{self.lr_scheduler_type}_{self.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}'   

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
            wandb.watch(self.model, self.criterion, log="all", log_freq=100)
            avg_train_loss, avg_train_iou_loss = self._train_epoch(epoch)
            if (epoch + 1) % self.interver == 0:
                self.valid(epoch, avg_train_loss, avg_train_iou_loss)
            if self.scheduler != None:
                self.scheduler.step()
        print(f"Training completed. Best Dice Score: {self.best_dice:.4f}, Best IoU Score: {self.best_iou:.4f}")
        
class SwinTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config, sweep_mode):
        super().__init__(model, train_loader, val_loader, optimizer, criterion, config, sweep_mode)
        
        # 더 큰 accumulation steps
        self.accumulation_steps = config.get('accumulation_steps', 4)
        
        # Mixed Precision 필수
        self.mixed_precision = True
        
        # Gradient clipping 추가
        self.grad_clip_norm = 5.0

    def _get_default_swin_scheduler(self):
        """Swin Transformer 기본 scheduler 설정"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-5
       )

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_iou_loss = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        for step, (images, masks) in tqdm(enumerate(self.train_loader), 
                                        total=len(self.train_loader), 
                                        desc=f'Epoch {epoch+1}/{self.num_epochs}'):
            images = images.cuda().float()  # float 타입으로 변환
            masks = masks.cuda().float()    # float 타입으로 변환
            
            outputs = self.model(images)
            outputs = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
            
            # Loss 계산
            loss = self.criterion(outputs, masks)
            epoch_loss += loss.item()

            # IoU Loss 계산
            outputs = (torch.sigmoid(outputs) > 0.5).float()  # float 타입으로 변환
            epoch_iou_loss += (1.0 - iou(masks, outputs)).mean().item()
            
            self._backpropagate(loss, step)
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_iou_loss = epoch_iou_loss / len(self.train_loader)

        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_loss,
            "avg_iou_loss": avg_iou_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        })
        print(f"Epoch [{epoch+1}/{self.num_epochs}] - Avg Train Loss: {avg_loss:.4f}, Avg IoU Loss: {avg_iou_loss:.4f}")
            
        return avg_loss, avg_iou_loss

    def valid(self, epoch, avg_train_loss, avg_train_iou_loss):
        """
        Swin Transformer를 위한 validation 함수를 사용하도록 오버라이드
        """
        dice, IoU = validation_swin(
            epoch + 1, 
            self.model, 
            self.val_loader, 
            self.criterion, 
            self.classes,
            self.config
        )
        
        # WandB에 메트릭 로깅
        wandb.log({
            "epoch": epoch + 1,
            "dice_coefficient": dice,
            "iou": IoU
        })
        
        # Save CheckPoints
        if self.best_dice < dice:
            self._save_best_model(epoch, dice, IoU)

    def _backpropagate(self, loss, step):
        """
        기존 _backpropagate 메소드 사용
        """
        super()._backpropagate(loss, step)

    def _optimizer_step(self):
        if self.mixed_precision:
            # Gradient clipping 추가
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)