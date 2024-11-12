import os
import json

import numpy as np
import cv2
from sklearn.model_selection import GroupKFold

import torch
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(
            self,
            is_train=True,
            transforms=None,
            IMAGE_ROOT=None,
            LABEL_ROOT=None,
            CLASSES = []
        ):
        self.CLASSES = CLASSES
        self.CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
        self.is_train = is_train
        self.transforms = transforms
        self.img_root = IMAGE_ROOT
        self.label_root = LABEL_ROOT

        _filenames, _labelnames = self.load_filenames()       
        self.filenames, self.labelnames= self.split_filenames(_filenames,_labelnames)
    
    def split_filenames(self, _filenames, _labelnames, n=5):
        ''' split train-valid
        한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다. '''
        groups = [os.path.dirname(fname) for fname in _filenames]
        # dummy label
        ys = [0 for fname in _filenames]
        
        gkf = GroupKFold(n_splits=n)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                if i == 0:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                filenames = list(_filenames[y])
                labelnames = list(self._labelnames[y])
                                
        return filenames, labelnames
    
    def load_filenames(self):
        pngs = {
                os.path.relpath(os.path.join(root, fname), start=self.img_root)
                for root, _dirs, files in os.walk(self.img_root)
                for fname in files
                if os.path.splitext(fname)[1].lower() == ".png"
        }
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        

        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

        pngs = sorted(pngs)
        jsons = sorted(jsons)

        print(f"Loaded {len(pngs)} PNG files and {len(jsons)} JSON files.")
        return np.array(pngs), np.array(jsons)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(self.CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label