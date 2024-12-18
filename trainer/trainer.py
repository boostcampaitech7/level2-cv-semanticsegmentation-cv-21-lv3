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
        max_norm = 5
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

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
        wandb.log({
            "epoch": epoch + 1,
            "dice_coefficient": dice,
            "iou": IoU
        })
        
        # Save CheckPoints
        if self.best_dice < dice:
            print(f"Best performance at epoch: {epoch + 1}, {self.best_dice:.4f} -> {dice:.4f}")
            self.best_dice = dice
            self.best_iou = IoU
            # 모델 저장
            self.save_checkpoint()

    def save_checkpoint(self):
        """
        Save the model checkpoint.

        :param dice: The Dice score of the current epoch
        :param IoU: The IoU score of the current epoch
        """
        os.makedirs(self.save_dir, exist_ok=True)   
        if 'model_name' in self.config['model']:
            model_name = self.config['model']['model_name'].replace('/', '_')
        else:
            model_name = self.model_name.replace('/', '_')

        # 모델 저장 경로 생성
        save_name = f'{model_name}_{self.lr}_{self.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}'    

        # 모델 저장
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

        # WandB 초기화
        wandb.init(
            entity="ppp6131-yonsei-university",
            project="boostcamp7th_semantic_segmentation",
            config=self.config,
            group=self.model_name
        )

        # sweep 하이퍼파라미터 설정
        if self.sweep_mode:
            # self.num_epochs = wandb.config.num_epochs
            # self.batch_size = wandb.config.batch_size
            # self.optimizer.param_groups[0]['lr'] = wandb.config.learning_rate
            self.criterion._set_weight(wandb.config.dice_weight)
            dice_weight, bce_weight = self.criterion._get_weight()
            print(f"dice_weight:{dice_weight}, bce_weight:{bce_weight}")
            #print(f"lr:{self.optimizer.param_groups[0]['lr']}, batch_size:{self.batch_size}")

        # 저장될 모델 이름 설정
        self.lr_scheduler_type = self.config.get("lr_scheduler", {}).get("type", "default") 
        self.save_name = f'{self.model_name}_{self.lr}_{self.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}size_down ' \
                         if not 'lr_scheduler' in self.config else \
                         f'{self.model_name}_{self.lr_scheduler_type}_{self.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}'   

        # 저장될 모델 이름 Wandb에 적용
        wandb.run.name = self.save_name

        # 학습 수행
        for epoch in range(self.num_epochs):
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
        
      
class BeitTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config, sweep_mode):
        super().__init__(model, train_loader, val_loader, optimizer, criterion, config, sweep_mode)
        
        # BEiT 특화 설정
        self.mixed_precision = True
        self.accumulation_steps = config.get('accumulation_steps', 4)
        
        # 메모리 최적화 설정
        torch.backends.cudnn.benchmark = True  #속도 향상
        
        if torch.cuda.is_available():
            # 더 빠른 DataLoader
            self.train_loader.pin_memory = True
            self.val_loader.pin_memory = True
        
        # DICE loss를 위한 설정
        self.use_dice = True
        
        if not self.sweep_mode and 'lr_scheduler' not in config:
            self.scheduler = self._get_default_beit_scheduler()

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_dice_score = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        for step, (images, masks) in tqdm(enumerate(self.train_loader), 
                                        total=len(self.train_loader), 
                                        desc=f'Epoch {epoch+1}/{self.num_epochs}'):
            images = images.cuda().float()
            masks = masks.cuda().float()
            
            # Forward pass
            outputs = self.model(images)
            
            # Loss 계산 - BeitDiceLoss가 auxiliary loss 처리
            loss = self.criterion(outputs, masks)
            
            # Metrics 계산
            with torch.no_grad():
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                outputs = torch.sigmoid(outputs)
                dice_score = 1 - self.criterion.compute_dice_loss(outputs, masks)
                epoch_dice_score += dice_score.item()
            
            epoch_loss += loss.item()
            self._backpropagate(loss, step)
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_dice_score = epoch_dice_score / len(self.train_loader)
        
        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_dice_score": avg_dice_score,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        })
        
        return avg_loss, avg_dice_score
    
class HRNetTrainer(Trainer):
   """
   HRNet + OCR을 위한 Trainer Class
   기본 Trainer에서 HRNet 특화 설정 추가
   """
   def __init__(self, model, train_loader, val_loader, optimizer, criterion, config, sweep_mode):
       super().__init__(model, train_loader, val_loader, optimizer, criterion, config, sweep_mode)
       
       # HRNet 특화 설정
       hrnet_config = config.get('hrnet_config', {})
       self.aux_weight = hrnet_config.get('aux_weight', 0.4)
       
       # Mixed Precision 적용
       self.mixed_precision = True
       
       # Gradient Accumulation steps 설정
       self.accumulation_steps = config.get('accumulation_steps', 2)
       
       # Learning rate 설정
       if not self.sweep_mode and 'lr_scheduler' not in config:
           self.scheduler = self._get_default_hrnet_scheduler()

   def _get_default_hrnet_scheduler(self):
       """HRNet default scheduler 설정"""
       from torch.optim.lr_scheduler import CosineAnnealingLR
       return CosineAnnealingLR(
           self.optimizer,
           T_max=self.num_epochs,
           eta_min=1e-6
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
           images = images.cuda().float()
           masks = masks.cuda().float()
           
           # Forward pass
           with torch.cuda.amp.autocast(enabled=self.mixed_precision):
               outputs = self.model(images)
               
               # HRNet의 출력 처리 (main output + auxiliary output)
               if isinstance(outputs, dict):
                   loss = self.criterion(outputs, masks)  # HRNetOCRLoss가 처리
                   outputs = outputs['out']  # IoU 계산은 main output으로
               else:
                   loss = self.criterion(outputs, masks)
           
           epoch_loss += loss.item()
           
           # IoU Loss 계산 (main output 사용)
           with torch.no_grad():
               pred_masks = (torch.sigmoid(outputs) > 0.5).float()
               epoch_iou_loss += (1.0 - iou(masks, pred_masks)).mean().item()
           
           # Backpropagation
           self._backpropagate(loss, step)
       
       avg_loss = epoch_loss / len(self.train_loader)
       avg_iou_loss = epoch_iou_loss / len(self.train_loader)
       
       # Logging
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
       Gradient accumulation을 포함한 역전파
       """
       if self.mixed_precision:
           self.scaler.scale(loss / self.accumulation_steps).backward()
           if (step + 1) % self.accumulation_steps == 0:
               self.scaler.step(self.optimizer)
               self.scaler.update()
               self.optimizer.zero_grad()
       else:
           (loss / self.accumulation_steps).backward()
           if (step + 1) % self.accumulation_steps == 0:
               self.optimizer.step()
               self.optimizer.zero_grad()

   def valid(self, epoch, avg_train_loss, avg_train_iou_loss):
       """
       Validation with metrics logging
       """
       dice, IoU = validation(epoch + 1, self.model, self.val_loader, 
                            self.criterion, self.classes)
       
       # Logging
       wandb.log({
           "epoch": epoch + 1,
           "dice_coefficient": dice,
           "iou": IoU
       })
       
       # Save best model
       if self.best_dice < dice:
           print(f"Best performance at epoch: {epoch + 1}, {self.best_dice:.4f} -> {dice:.4f}")
           self.best_dice = dice
           self.best_iou = IoU
           self.save_checkpoint(dice, IoU)

   def _optimizer_step(self):
       """
       메모리 효율을 위한 최적화 단계
       """
       if self.mixed_precision:
           self.scaler.step(self.optimizer)
           self.scaler.update()
       else:
           self.optimizer.step()
       
       self.optimizer.zero_grad(set_to_none=True)
       
       if torch.cuda.is_available():
           torch.cuda.empty_cache()