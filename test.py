import os
import argparse
import time
import tqdm
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


from utils.util import encode_mask_to_rle, load_config
from utils.dataset import XRayInferenceDataset, SwinXRayInferenceDataset, BeitXRayInferenceDataset, HRNetXRayInferenceDataset  # 수정된 import

<<<<<<< HEAD
def tta_augmentations():
    """
    Define Test-Time Augmentations (TTA) using Albumentations.
    """
    return [
        A.Compose([A.Resize(1024, 1024)]),
        A.Compose([A.Resize(1024, 1024), A.CLAHE(clip_limit=(2, 4), tile_grid_size=(8, 8))]),
        A.Compose([A.Resize(1024, 1024), A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5)]),
        A.Compose([A.Resize(1024, 1024), A.RandomBrightnessContrast(brightness_limit=0.32, contrast_limit=0.1, p=0.7)]),
        A.Compose([A.Resize(1024, 1024), A.Rotate(limit=10, p=0.5)]),
        A.Compose([A.Resize(1024, 1024), A.HorizontalFlip(p=0.5)]),
    ]

def tta_test(model, data_loader, classes, tta_transforms, thr=0.5):
    CLASS2IND = {v: i for i, v in enumerate(classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(classes)

        for step, (images, image_names) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            all_outputs = []
            images = images.permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, C]로 변환 후 numpy 배열로 변경

            for tta in tta_transforms:
                # Apply TTA
                augmented_images = [tta(image=image)["image"] for image in images]
                augmented_images = torch.tensor(np.stack(augmented_images)).permute(0, 3, 1, 2).cuda()  # [N, C, H, W]로 복구

                # Predict
                outputs = model(augmented_images)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']

                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                all_outputs.append(outputs.detach().cpu().numpy())

            # Average predictions across TTA
            averaged_outputs = np.mean(all_outputs, axis=0)
            averaged_outputs = (averaged_outputs > thr).astype(np.uint8)
            
            for output, image_name in zip(averaged_outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def save_csv(rles, filename_and_class,model_name='output', save_root='./output/'):
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(f"{save_root}{model_name}.csv", index=False)
=======
def save_csv(rles, filename_and_class, save_root='./output/'):
   # 기존 코드 유지
   classes, filename = zip(*[x.split("_") for x in filename_and_class])
   image_name = [os.path.basename(f) for f in filename]
   
   df = pd.DataFrame({
       "image_name": image_name,
       "class": classes,
       "rle": rles,
   })
   
   df.to_csv(f"{save_root}output{time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)
>>>>>>> Ahn-latte/issue16

def test(model, data_loader, classes, thr=0.5):
   # 기존 코드
   CLASS2IND = {v: i for i, v in enumerate(classes)}
   IND2CLASS = {v: k for k, v in CLASS2IND.items()}

   model = model.cuda()
   model.eval()

   rles = []
   filename_and_class = []
   with torch.no_grad():
       n_class = len(classes)

<<<<<<< HEAD
        for step, (images, image_names) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class
=======
       for step, (images, image_names) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
           images = images.cuda()    
           outputs = model(images)
           if isinstance(outputs, dict) and 'out' in outputs:
               outputs = outputs['out']
           
           # 기존 코드
           # outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
           
           # 수정된 코드 - config에서 interpolation 설정 가져오기
           outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear", align_corners=False)
           outputs = torch.sigmoid(outputs)
           outputs = (outputs > thr).detach().cpu().numpy()
           
           for output, image_name in zip(outputs, image_names):
               for c, segm in enumerate(output):
                   rle = encode_mask_to_rle(segm)
                   rles.append(rle)
                   filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                   
   return rles, filename_and_class
>>>>>>> Ahn-latte/issue16

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    args = parser.parse_args()
   
    config = load_config('/data/ephemeral/home/ahn/config/config.json')
    model_path = args.model_path 
    model_name = model_path.split('.')[1].split('/')[2]
    print(model_name)
    os.path.exists(model_path)

    model = torch.load(model_path)
    classes = config['CLASSES']

<<<<<<< HEAD
    tf = A.Resize(1024,1024)
    test_dataset = XRayInferenceDataset(
        transforms=tf,
        img_root=f"{config['DATA_ROOT']}/test/DCM",
        classes=classes)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    rles, filename_and_class = test(model, test_loader,classes)
    # tta_transforms = tta_augmentations()
    # rles, filename_and_class = tta_test(model, test_loader, classes, tta_transforms)
    save_csv(rles, filename_and_class, model_name)
=======
   # 기존 코드
   # tf = A.Resize(512, 512)
   # test_dataset = XRayInferenceDataset(
   #     transforms=tf,
   #     img_root=f"{config['DATA_ROOT']}/test/DCM",
   #     classes=classes)
   
   # 수정된 코드
    tf = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(transpose_mask=False)  # [H, W, C] -> [C, H, W]로 변환
])

   # Dataset 선택
    Dataset = HRNetXRayInferenceDataset
    test_dataset = Dataset(
       transforms=tf,
       img_root=f"{config['DATA_ROOT']}/test/DCM",
       classes=classes,
       config=config
   )

   # 수정된 코드
    test_loader = DataLoader(
       dataset=test_dataset, 
       batch_size=4,  # SwinV2-L을 위한 작은 배치 사이즈
       shuffle=False,
       num_workers=4,  # 메모리 효율을 위해 감소
       pin_memory=True,
       drop_last=False
   )
   
    rles, filename_and_class = test(model, test_loader, classes)
    save_csv(rles, filename_and_class)
>>>>>>> Ahn-latte/issue16

if __name__ == '__main__':
   main()