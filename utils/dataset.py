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
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"]
            
        return image, label
    
class SwinXRayDataset(Dataset):
    def __init__(
            self,
            is_train=True,
            transforms=None,
            img_root=None,
            label_root=None,
            classes=[],
            config=None  # Swin config 추가
        ):
        self.classes = classes
        self.CLASS2IND = {v: i for i, v in enumerate(classes)}
        self.is_train = is_train
        self.transforms = transforms
        self.img_root = img_root
        self.label_root = label_root
        
        # Swin config에서 필요한 설정 가져오기
        self.img_size = config.get('img_size', 224)  # Swin default
        
        _filenames, _labelnames = self.load_filenames()       
        self.filenames, self.labelnames= self.split_filenames(_filenames,_labelnames)
    
    def split_filenames(self, _filenames, _labelnames, n=5):
        groups = [os.path.dirname(fname) for fname in _filenames]
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
        # Swin 입력 크기에 맞게 조정
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        label_shape = (self.img_size, self.img_size, len(self.classes))
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            # 포인트 좌표를 새로운 이미지 크기에 맞게 조정
            points = points * (self.img_size / image.shape[0])
            points = points.astype(np.int32)
            
            class_label = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"]
        
        return image, label

class SwinXRayInferenceDataset(Dataset):
    def __init__(self, 
                 transforms=None,
                 img_root=None,
                 classes=[],
                 config=None  # Swin config 추가
                 ):
        self.classes = classes
        self.img_root = img_root
        self.transforms = transforms
        self.img_size = config.get('img_size', 224)  # Swin default
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
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)  
        image = torch.from_numpy(image).float()
            
        return image, image_name
    
class BeitXRayDataset(Dataset):
    def __init__(
            self,
            is_train=True,
            transforms=None,
            img_root=None,
            label_root=None,
            classes=[],
            config=None
        ):
        super().__init__()
        self.classes = classes
        self.CLASS2IND = {v: i for i, v in enumerate(classes)}
        self.is_train = is_train
        self.transforms = transforms
        self.img_root = img_root
        self.label_root = label_root
        
        # Size 설정
        self.original_size = config['model'].get('original_size', 2048)
        self.model_size = config['model'].get('model_size', 640)
        
        _filenames, _labelnames = self.load_filenames()       
        self.filenames, self.labelnames = self.split_filenames(_filenames, _labelnames)
        
        print(f"\nDataset Info:")
        print(f"Mode: {'Train' if is_train else 'Valid'}")
        print(f"Original size: {self.original_size}x{self.original_size}")
        print(f"Model input size: {self.model_size}x{self.model_size}")

    def load_filenames(self):
        """파일 목록 로드 및 검증"""
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
        
        # 파일 매칭 검증
        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
        
        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0, "Missing image files"
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0, "Missing annotation files"
        
        return np.array(sorted(pngs)), np.array(sorted(jsons))

    def split_filenames(self, _filenames, _labelnames, n=5):
        """데이터셋 분할"""
        # 폴더별 그룹화
        groups = [os.path.dirname(fname) for fname in _filenames]
        unique_groups = set(groups)
        print(f"\nSplit Info:")
        print(f"Total groups: {len(unique_groups)}")
        
        ys = [0 for _ in _filenames]  # dummy labels
        
        # GroupKFold로 분할
        gkf = GroupKFold(n_splits=n)
        
        filenames = []
        labelnames = []
        
        # 분할 수행
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                if i == 0:  # 첫 번째 fold는 validation
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break
        
        print(f"Split ratio: {n}-fold")
        print(f"Selected samples: {len(filenames)}")
        return filenames, labelnames

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_root, image_name)
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 라벨 처리
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # 라벨 생성 (원본 크기)
        label_shape = (self.original_size, self.original_size, len(self.classes))
        label = np.zeros(label_shape, dtype=np.float32)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        # 원본 크기로 마스크 생성
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            mask = np.zeros((self.original_size, self.original_size), dtype=np.float32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            label[..., class_ind] = mask
        
        # Transform 적용 (resize 포함)
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"]
        
        return image, label

class BeitXRayInferenceDataset(Dataset):
    def __init__(
        self,
        transforms=None,
        img_root=None,
        classes=[],
        config=None
    ):
        self.classes = classes
        self.img_root = img_root
        self.transforms = transforms
        self.model_size = config['model'].get('img_size', 640) if config else 640
        self.filenames = self.load_filenames()
        
        print(f"\nDataset Info:")
        print(f"Number of test samples: {len(self.filenames)}")
        print(f"Model input size: {self.model_size}x{self.model_size}")

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
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # resize
        image = cv2.resize(image, (self.model_size, self.model_size))
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            
        # permute을 통한 채널 순서 조정
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            
        # 첫 번째 아이템에 대해 디버그 정보 출력
        if item == 0:
            print(f"\nFirst item info:")
            print(f"Image shape: {image.shape}")
            print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
            
        return image, image_name
    
class HRNetXRayDataset(Dataset):
    def __init__(
            self,
            is_train=True,
            transforms=None,
            img_root=None,
            label_root=None,
            classes=[],
            config=None
        ):
        self.classes = classes
        self.CLASS2IND = {v: i for i, v in enumerate(classes)}
        self.is_train = is_train
        self.transforms = transforms
        self.img_root = img_root
        self.label_root = label_root
        
        # HRNet 설정
        self.img_size = config['model'].get('img_size', 512)  # HRNet 기본 입력 크기
        
        _filenames, _labelnames = self.load_filenames()       
        self.filenames, self.labelnames = self.split_filenames(_filenames, _labelnames)
        
        print(f"\nDataset Info:")
        print(f"Mode: {'Train' if is_train else 'Valid'}")
        print(f"Number of samples: {len(self.filenames)}")
        print(f"Image size: {self.img_size}x{self.img_size}")

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_root, image_name)
        
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 리사이즈
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # 라벨 생성 - 이미 리사이즈된 이미지 크기 사용
        label_shape = (self.img_size, self.img_size, len(self.classes))
        label = np.zeros(label_shape, dtype=np.float32)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
            
        original_size = image.shape[0]  # 원본 이미지 크기로 수정
        
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            # 포인트 좌표 조정 - 수정된 부분
            points = points * (self.img_size / 2048)  # 원본 크기(2048)에서 변환
            points = points.astype(np.int32)
            
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            cv2.fillPoly(mask, [points], 1)
            label[..., class_ind] = mask
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"]
        
        return image, label
    
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

class HRNetXRayInferenceDataset(Dataset):
    def __init__(
        self,
        transforms=None,
        img_root=None,
        classes=[],
        config=None
    ):
        self.classes = classes
        self.img_root = img_root
        self.transforms = transforms
        self.img_size = config['model'].get('img_size', 512)
        self.filenames = self.load_filenames()
        
        print(f"\nDataset Info:")
        print(f"Number of test samples: {len(self.filenames)}")
        print(f"Input image size: {self.img_size}x{self.img_size}")

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
        
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # transforms 적용
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
            
        # Debug info for first item
        if item == 0:
            print(f"\nFirst item info:")
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            if isinstance(image, torch.Tensor):
                print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
        
        return image, image_name