import os
import json
import numpy as np
import cv2
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from sklearn.model_selection import train_test_split

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


def get_data(data_root, split_ratio=0.1):
    global IMAGE_ROOT, LABEL_ROOT
    IMAGE_ROOT = os.path.join(data_root, "DCM")
    LABEL_ROOT = os.path.join(data_root, "outputs_json")

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    all_filenames = list(zip(sorted(pngs), sorted(jsons)))
    train_data, val_data = train_test_split(all_filenames, test_size=split_ratio, random_state=7)

    return train_data, val_data
@DATASETS.register_module()
class XRayboneDataset(BaseSegDataset):
    def __init__(self, data_root, is_train, **kwargs):
        self.is_train = is_train
        self.data_root = data_root
        super().__init__(**kwargs,data_root=data_root)
        
    def load_data_list(self):
        train_data, val_data = get_data(data_root=self.data_root, split_ratio= 0.1)
        data_pairs = train_data if self.is_train else val_data

        
        data_list = []
        for img_path, ann_path in data_pairs:
            data_info = dict(
                img_path=os.path.join(IMAGE_ROOT, img_path),
                seg_map_path=os.path.join(LABEL_ROOT, ann_path),
            )
            data_list.append(data_info)
        
        return data_list
