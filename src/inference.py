import numpy as np 
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision

import os, time, glob, argparse, cv2
from transform import *

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pred_rle(filename, model, predictions):
    image = cv2.imread(filename, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = image.shape[:2]
    image = cv2.resize(image, (256,256), cv2.INTER_AREA)
    preds = model(infer_transform(image).unsqueeze(0).to(DEVICE))['out'][0]
    for i, mask in enumerate(preds):
        if (mask > 0).float().sum() > 0:
            mask = cv2.resize((mask.detach().cpu() > 0).numpy().astype(np.uint8), size, cv2.INTER_AREA)
            predictions.append([filename.split('/')[-1], rle_encode(mask), i])
        
def inference():
    filename = glob.glob(os.path.join(args.data_path, '*.jpg'))[:10]
    
    if args.model_name == 'deeplabv3_resnet50':
        model = torchvision.models.segmentation.deeplabv3_resnet50(False)
    else:
        model = torchvision.models.segmentation.fcn_resnet50(False)
        
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 46, 1)
    
    state_dict = torch.load(args.pht_file, map_location='cpu')
    model.load_state_dict(state_dict)
    
    model = model.to(DEVICE)
    
    predictions = []
    for file in filename:
        pred_rle(file, model, predictions)
        
    df = pd.DataFrame(predictions, columns=['ImageId','EncodedPixels','ClassId'])
    df.to_csv(args.csv_file, index=False)
    
parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_name', default='fcn_resnet50', type=str,
                    help='Segmentation model name: fcn_resnet50 or deeplabv3_resnet50')
# data and log files
parser.add_argument('--data_path', default='data/test', type=str, help='Path to image data')
parser.add_argument('--pht_file', default='checkpoint/fcn_resnet50.pth', type=str, help='Name for saved weights file')
parser.add_argument('--csv_file', default='submission.csv', type=str, help='CSV file for saving rle strings')

args = parser.parse_args()

if __name__ == '__main__':
    
    DEVICE = xm.xla_device()
    inference()
    
    
    