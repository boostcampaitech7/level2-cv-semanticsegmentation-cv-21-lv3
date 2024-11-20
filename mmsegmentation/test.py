# 필요한 라이브러리 임포트
import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from prettytable import PrettyTable
from torch import nn
from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmseg.datasets import BaseSegDataset
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.decode_heads import ASPPHead, FCNHead, SegformerHead 
from mmseg.models.utils.wrappers import resize
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner, load_checkpoint
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.structures import PixelData
from mmcv.transforms import BaseTransform
from mmseg.apis import init_model
from collections import defaultdict
import argparse

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()

# 모델 초기화
config_path = args.config
checkpoint_path = args.checkpoint
cfg = Config.fromfile(config_path)
cfg.launcher = "none"
model = init_model(config_path, checkpoint_path)

# 경로 설정
IMAGE_ROOT = "/home/minipin/level2-cv-semanticsegmentation-cv-21-lv3/data/test/DCM"
checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
SAVE_DIR = os.path.join("/home/minipin/level2-cv-semanticsegmentation-cv-21-lv3/mmsegmentation/predictions", checkpoint_name)
os.makedirs(SAVE_DIR, exist_ok=True)

# 클래스 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

# 데이터 준비 함수
def _preprare_data(imgs, model):
    pipeline = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='PackSegInputs')
    ])
    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False
    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'
    data = defaultdict(list)
    for img in imgs:
        data_ = dict(img=img) if isinstance(img, np.ndarray) else dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])
    return data, is_batch

# 색상 정의
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 레이블을 RGB 이미지로 변환
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
    return image

# PNG 파일 목록 생성
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

# 데이터셋 클래스 정의
class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = np.array(sorted(pngs))
        self.filenames = _filenames
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path)
        return image, image_name, image_path

# 마스크를 RLE로 인코딩
def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE를 마스크로 디코딩
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(height, width)

# 테스트 함수
def test(model, data_loader, thr=0.5):
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)
        for step, (images, image_names, image_path) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = image_path
            data, is_batch = _preprare_data(images, model)
            results = model.test_step(data)
            results = results[0].pred_sem_seg.data
            results = (results > thr).detach().cpu().numpy().astype(np.uint8).reshape(1, 29, 2048, 2048)
            for output, image_name in zip(results, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    return rles, filename_and_class

# 데이터 로더 설정
test_dataset = XRayInferenceDataset()
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False
)

# 테스트 실행 및 결과 저장
rles, filename_and_class = test(model, test_loader)
classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]
df = pd.DataFrame({"image_name": image_name, "class": classes, "rle": rles})
df.to_csv(os.path.join(SAVE_DIR, "output.csv"), index=False)
print(f"Predictions saved to {SAVE_DIR}/output.csv")