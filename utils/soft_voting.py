import os
import argparse

import tqdm
import pandas as pd
import albumentations as A

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.util import encode_mask_to_rle, load_config
from utils.dataset import XRayInferenceDataset
import numpy as np

def soft_vote_test(model_dir, data_loader, classes, thr=0.5):
    """
    Perform soft voting among multiple models for test-time inference.
    """
    CLASS2IND = {v: i for i, v in enumerate(classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    model_paths = [
        os.path.join(model_dir, f) 
        for f in os.listdir(model_dir) 
        if f.endswith('.pt')
    ]

    print(model_paths)

    models = [torch.load(path).cuda().eval() for path in model_paths]

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(classes)

        for step, (images, image_names) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            all_outputs = []

            for model in models:
                outputs = model(images)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']
                
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                all_outputs.append(outputs.detach().cpu().numpy())

            # Average predictions across models (soft voting)
            averaged_outputs = np.mean(all_outputs, axis=0)
            averaged_outputs = (averaged_outputs > thr).astype(np.uint8)
            
            for output, image_name in zip(averaged_outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def save_csv(rles, filename_and_class, save_root='./output/'):
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(f"{save_root}output.csv", index=False)

def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, nargs='+', required=True, help='Paths to the model files')
    args = parser.parse_args()
    
    config = load_config('./config/config.json')
    model_dir = args.model_dir
    for path in model_dir:
        assert os.path.exists(path), f"Model dir {path} does not exist."

    classes = config['CLASSES']

    tf = A.Resize(1024, 1024)
    test_dataset = XRayInferenceDataset(
        transforms=tf,
        img_root=f"{config['DATA_ROOT']}/test/DCM",
        classes=classes)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # Perform soft voting with multiple models
    rles, filename_and_class = soft_vote_test(model_dir, test_loader, classes)
    save_csv(rles, filename_and_class)

if __name__ == '__main__':
    main()
