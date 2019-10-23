import numpy as np 
import pandas as pd
import logging, json

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision

import os, time, random, argparse

from dataset import FashionDataset
from transform import *
import schedulers

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm

def set_seeds(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def infer_loop_fn(model, loader, device, context):
    
    loss_fn = nn.BCEWithLogitsLoss()
    score = []
    model.eval()
    for x, (data, target) in loader:
        output = model(data)
        loss = loss_fn(output, target)
        score.append(loss.item())
        if (args.log_step) and (x % args.log_step) == 0:
            logging.info('[{}]({}) Loss={:.4f}'.format(device, x, loss.item()))
            
    score = sum(score) / len(score)
    return score

def train_loop_fn(model, loader, device, context):
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.FloatTensor([7]).to(device))
    optimizer = context.getattr_or(
      'optimizer',
      lambda: torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-08,
                                betas=(0.9, 0.999), weight_decay=args.weight_decay) 
    )

    lr_scheduler = context.getattr_or(
        'lr_scheduler', lambda: schedulers.wrap_optimizer_with_scheduler(
            optimizer,
            scheduler_type='WarmupAndExponentialDecayScheduler',
            scheduler_divisor=args.slr_divisor,
            scheduler_divide_every_n_epochs=args.slr_divide_n_epochs,
            num_warmup_epochs=args.num_warmup_epochs,
            min_delta_to_update_lr=args.num_warmup_epochs,
            num_steps_per_epoch=num_steps_per_epoch))
    
    score = []
    model.train()
    for x, (data, target) in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output['out'], target)
        loss.backward()
        xm.optimizer_step(optimizer)
        score.append(loss.item())
        if (args.log_step) and (x % args.log_step) == 0:
            logging.info('[{}]({}) Loss={:.4f}'.format(device, x, loss.item() ))
        if lr_scheduler:
            lr_scheduler.step()

    score = sum(score) / len(score)
    return score

def train():
    set_seeds()
    logging.info('Loading masks...')
    with open(args.json_file,'r') as f:
        masks = json.load(f)
        
    # for example use only 200 images    
    filename = list(masks.keys())[:200]
    
    global devices, num_steps_per_epoch
    
    devices = (
        xm.get_xla_supported_devices(max_devices=args.num_cores) if args.num_cores != 0 else [])

    logging.info('Start training model')
    if args.model_name == 'deeplabv3_resnet50':
        m = torchvision.models.segmentation.deeplabv3_resnet50(False)
    else:
        m = torchvision.models.segmentation.fcn_resnet50(False)
        
    m.classifier[-1] = torch.nn.Conv2d(m.classifier[-1].in_channels, 46, 1)
    # wrapped for parallel training
    model = dp.DataParallel(m,  device_ids=devices)
    
    ds = FashionDataset(filename, masks, path=args.data_path,
                        transform=train_transform, size=(256,256))
    loader = D.DataLoader(ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_worker)

    num_steps_per_epoch = len(loader) // len(devices)

    for epoch in range(1, args.epochs + 1):
        
        train_loss = model(train_loop_fn, loader)
        train_loss = np.array(train_loss).mean()
        
        logging.info('[Epoch {:3d}] Train loss: {:.3f}'.format(epoch, train_loss))

    # Save weights
    state_dict = model.models[0].to('cpu').state_dict()
    torch.save(state_dict, args.save_file)
    logging.info('')
    logging.info('Model saved\n')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', default='fcn_resnet50', type=str,
                    help='Segmentation model name: fcn_resnet50 or deeplabv3_resnet50')

parser.add_argument('--epochs', default=2, type=int, help='Number of training epochs')
parser.add_argument('--log_step', default=100, type=int, help='Report every N steps loss, 0 - disable')

parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_cores', default=8, type=int, help='Number of cores TPU')
parser.add_argument('--num_worker', default=4, type=int, help='Number of CPU core for loader')

# optimizer params
parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for AdamW')

# scheduler params
parser.add_argument('--slr_divisor', default=2, type=float, help='Scheduler divisor')
parser.add_argument('--slr_divide_n_epochs', default=5, type=float, help='Scheduler divide every N epochs')
parser.add_argument('--num_warmup_epochs', default=0.5, type=float, help='Warmup epoch')
parser.add_argument('--min_lr', default=1e-5, type=float, help='Min learning rate')

            
# data and log files
parser.add_argument('--data_path', default='data/train', type=str, help='Path to image data')
parser.add_argument('--json_file', default='data/train.json', type=str, help='JSON file with rle mask')
parser.add_argument('--log_file', default='report/train.xla.log', type=str, help='Name for log file')
parser.add_argument('--save_file', default='checkpoint/model.pth', type=str, help='Name for saved weights file')

args = parser.parse_args()

if __name__ == '__main__':

    logging.basicConfig(filename=args.log_file, filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        level=logging.INFO)
    train()
    
    
    