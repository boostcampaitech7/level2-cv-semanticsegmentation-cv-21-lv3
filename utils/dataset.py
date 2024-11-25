import os
import json

import numpy as np
import cv2
from sklearn.model_selection import GroupKFold

import torch
from torch.utils.data import Dataset


class XRayInferenceDataset(Dataset):
    def __init__(self, 
                 transforms=None,
                 img_root=None,
                 classes = []
                 ):
        self.classes = classes
        self.img_root = img_root
        self.transforms = transforms
        self.filenames = self.load_filenames()
        
    def load_filenames(self):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.img_root)
            for root, _dirs, files in os.walk(self.img_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        return np.array(sorted(pngs))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name

class XRayDataset(Dataset):
    def __init__(
            self,
            is_train=True,
            transforms=None,
            img_root=None,
            label_root=None,
            classes = []
        ):
        self.classes = classes
        self.CLASS2IND = {v: i for i, v in enumerate(classes)}
        self.is_train = is_train
        self.transforms = transforms
        self.img_root = img_root
        self.label_root = label_root

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
                labelnames = list(_labelnames[y])
                                
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
        label_shape = tuple(image.shape[:2]) + (len(self.classes), )
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
            
        # self.save_visualization(image, label, image_name)
        # self.save_image_visualization(image,image_name)
    
        return image, label

    # def __getitem__(self, item):
    #     image_name = self.filenames[item]
    #     image_path = os.path.join(self.img_root, image_name)
        
    #     image = cv2.imread(image_path)
    #     image = image / 255.
        
    #     label_name = self.labelnames[item]
    #     label_path = os.path.join(self.label_root, label_name)
        
    #     # label 파일 읽기
    #     with open(label_path, "r") as f:
    #         annotations = json.load(f)
    #     annotations = annotations["annotations"]
        
    #     if self.is_train:
    #         # 모든 annotation의 points를 고려하여 가장 작은 x_min, y_min과 가장 큰 x_max, y_max 찾기
    #         x_min, y_min = float('inf'), float('inf')
    #         x_max, y_max = float('-inf'), float('-inf')
    #         for ann in annotations:
    #             points = np.array(ann["points"])
    #             x_min = min(x_min, np.min(points[:, 0]))
    #             y_min = min(y_min, np.min(points[:, 1]))
    #             x_max = max(x_max, np.max(points[:, 0]))
    #             y_max = max(y_max, np.max(points[:, 1]))
            
    #         # 정사각형으로 만들기
    #         center_x = (x_min + x_max) / 2
    #         center_y = (y_min + y_max) / 2
    #         side_length = max(x_max - x_min, y_max - y_min)
    #         bbox = [
    #             int(center_x - side_length / 2),
    #             int(center_y - side_length / 2),
    #             int(center_x + side_length / 2),
    #             int(center_y + side_length / 2)
    #         ]
            
    #         # Bounding box 확장
    #         expand = 50
    #         bbox[0] = max(0, bbox[0] - expand)
    #         bbox[1] = max(0, bbox[1] - expand)
    #         bbox[2] = min(image.shape[1], bbox[2] + expand)
    #         bbox[3] = min(image.shape[0], bbox[3] + expand)
            

    #         # 이미지와 label crop
    #         image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    #         cropped_label = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], len(self.classes)), dtype=np.uint8)
    #         for ann in annotations:
    #             c = ann["label"]
    #             class_ind = self.CLASS2IND[c]
    #             points = np.array(ann["points"]) - [bbox[0], bbox[1]]  # 좌표 이동
    #             class_label = np.zeros(cropped_label.shape[:2], dtype=np.uint8)
    #             cv2.fillPoly(class_label, [points], 1)
    #             cropped_label[..., class_ind] = class_label
    #         label = cropped_label
    #     else:
    #         # Validation 단계에서는 원본 이미지를 그대로 사용
    #         label = np.zeros(image.shape[:2] + (len(self.classes),), dtype=np.uint8)
    #         for ann in annotations:
    #             c = ann["label"]
    #             class_ind = self.CLASS2IND[c]
    #             points = np.array(ann["points"])
    #             class_label = np.zeros(image.shape[:2], dtype=np.uint8)
    #             cv2.fillPoly(class_label, [points], 1)
    #             label[..., class_ind] = class_label
        
    #     if self.transforms is not None:
    #         inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
    #         result = self.transforms(**inputs)
            
    #         image = result["image"]
    #         label = result["mask"] if self.is_train else label

    #     # Tensor 변환
    #     image = image.transpose(2, 0, 1)  # channel first
    #     label = label.transpose(2, 0, 1)

    #     image = torch.from_numpy(image).float()
    #     label = torch.from_numpy(label).float()
        
    #     # self.save_visualization(image, label, image_name)

    #     return image, label
    
    def save_visualization(self, image, label, image_name):
        # 이미지와 label을 시각화하여 저장
        image_np = image.numpy().transpose(1, 2, 0) * 255  # channel last로 변환
        image_np = image_np.astype(np.uint8)
        
        # 각 클래스에 대해 다른 색상으로 label을 시각화
        for i in range(label.shape[0]):
            mask = label[i].numpy()
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            image_np[mask > 0] = image_np[mask > 0] * 0.5 + np.array(color) * 0.5

        # 결과 이미지 저장
        save_path = os.path.join("visualizations_calcul", f"visual_{image_name}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image_np)
        
    def save_image_visualization(self, image, image_name):
        # 이미지 시각화하여 저장
        image_np = image.numpy().transpose(1, 2, 0) * 255  # channel last로 변환
        image_np = image_np.astype(np.uint8)

        # 결과 이미지 저장
        save_path = os.path.join("visualizations_image", f"visual_{image_name}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image_np)