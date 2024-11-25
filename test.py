import os
import argparse
import time
import tqdm
import pandas as pd
import albumentations as A

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.util import encode_mask_to_rle, load_config
from utils.dataset import XRayInferenceDataset, SwinXRayInferenceDataset  # 수정된 import

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    args = parser.parse_args()
   
    config = load_config('/data/ephemeral/home/ahn/config/config.json')
    model_path = args.model_path 
    os.path.exists(model_path)

    model = torch.load(model_path)
    classes = config['CLASSES']

   # 기존 코드
   # tf = A.Resize(512, 512)
   # test_dataset = XRayInferenceDataset(
   #     transforms=tf,
   #     img_root=f"{config['DATA_ROOT']}/test/DCM",
   #     classes=classes)
   
   # 수정된 코드
    tf = A.Compose([
       A.Resize(config['model'].get('img_size', 384), config['model'].get('img_size', 384)),
       A.Normalize()
   ])

   # Dataset 선택
    Dataset = SwinXRayInferenceDataset if config['model']['type'] == 'timm' else XRayInferenceDataset
    test_dataset = Dataset(
       transforms=tf,
       img_root=f"{config['DATA_ROOT']}/test/DCM",
       classes=classes,
       config=config
   )

   # 수정된 코드
    test_loader = DataLoader(
       dataset=test_dataset, 
       batch_size=2,  # SwinV2-L을 위한 작은 배치 사이즈
       shuffle=False,
       num_workers=4,  # 메모리 효율을 위해 감소
       pin_memory=True,
       drop_last=False
   )
   
    rles, filename_and_class = test(model, test_loader, classes)
    save_csv(rles, filename_and_class)

if __name__ == '__main__':
   main()