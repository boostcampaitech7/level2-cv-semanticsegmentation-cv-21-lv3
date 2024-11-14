import os

import tqdm
import pandas as pd
import albumentations as A

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.util import encode_mask_to_rle, load_config
from utils.dataset import XRayInferenceDataset

def save_csv(rles, filename_and_class,save_root='./output/'):
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(f"{save_root}output.csv", index=False)

def test(model, data_loader, classes, thr=0.5):
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
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def main():
    config = load_config('./config/config.json')
    classes = config['CLASSES']
    model = torch.load(os.path.join(config['SAVED_DIR'], "fcn_resnet50_best_model.pt"))

    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(
        transforms=tf,
        img_root=config['TEST_IMAGE_ROOT'],
        classes=classes)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    rles, filename_and_class = test(model, test_loader,classes)
    save_csv(rles, filename_and_class)

if __name__ == '__main__':
    main()