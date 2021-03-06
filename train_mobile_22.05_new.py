#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


import torchvision
from torchvision import transforms

from loss import AngleLoss

from utils import printoneline, dt, save_model
from transforms.noising import RawNoise
from networks.faceid.mobile import MobileFacenet
from networks.faceid.mobile import ArcMarginProduct
from torch.optim import lr_scheduler
  
from common import CKPT_DIR, LOGS_DIR
import itertools

def freeze_model(model):
#     model.train(False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def prepare_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

from base import BaseExpRunner
class JointTrainer(BaseExpRunner):
    num_avg_batches = 2 # 64 # 32
    
    def global_forward(self, sample, batch_idx):
        optimizer.zero_grad()
        imgs, labels = sample
        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

    #     mosaic, groundtruth, labels = sample
    #     mosaic, groundtruth, labels = mosaic.cuda(non_blocking=True), groundtruth.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        if (batch_idx+1)%self.num_avg_batches == 0:
            make_step = True
        else:
            make_step = False

        if batch_idx%self.num_avg_batches == 0:
            zero_grad = True
        else:
            zero_grad = False

        imgs = (255*imgs - 127.5)/128
        # compute output
        if zero_grad:
            optimizer.zero_grad()
        raw_logits = faceid(imgs)
        outputs = ArcMargin(raw_logits, labels)
        faceid_loss = faceid_criterion(outputs, labels)
        loss = faceid_loss
        loss = loss/self.num_avg_batches
        loss.backward()

    #         torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 10)
    #         torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 10)
        if make_step:
            optimizer.step()

#         _, predicted = torch.max(raw_logits.data, 1)
#         total += labels.size(0)
#         correct += int(predicted.eq(labels.data).sum())

        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
        cur_grad_faceid_norm = np.sum(grads)

        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, ArcMargin.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
        cur_arcmargin_grad_norm = np.sum(grads)

        cur_faceid_loss = float(faceid_loss)
        self.tmp_logs_dict['faceid_loss'].append(cur_faceid_loss)

        self.total_loss += float(loss) # FIXME
        if batch_idx % 50 == 0:
            printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | faceid: %.4f grad: %.4f gradM: %.4f' % 
                         (self.cur_epoch, self.total_loss/(batch_idx+1), batch_idx, cur_faceid_loss, 
                         cur_grad_faceid_norm, cur_arcmargin_grad_norm))

            
# sig_read_linspace = np.linspace(-3,-1.5,4)
# sig_shot_linspace = np.linspace(-2,-1,4)

# sig_read = sig_read_linspace[3]
# sig_shot = sig_shot_linspace[2]
# a = np.power(10., sig_read)
# b = np.power(10., sig_shot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of the experiment')
    parser.add_argument('-d', '--device', type=str, required=True,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    
    if len(args.device)>1:
        multigpu_mode = True
        # SEE: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        # SEE: https://github.com/pytorch/examples/tree/master/dcgan
        # dist.init_process_group(backend="ncll", init_method="tcp://127.0.0.1:9988",
#                                 world_size=1, rank=0)
    else:
        multigpu_mode = False
        
    
    
    train_data_dir = "/tmp/CASIA-WebFace-sphereface"
#     train_data_dir = "/tmp/QMUL_96x112"

#     transform = transforms.Compose([
#                              transforms.ToTensor()
#                          ])
    a = 0.15
    b = 0.15
    transform = transforms.Compose([
                         transforms.CenterCrop((112,96)), #Random
                         transforms.RandomHorizontalFlip(),
#                          RawNoise(a, b, 0.7, 0.6),
                         transforms.ToTensor()
                     ])
    dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
    dataloader_train = torch.utils.data.dataloader.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=14)
    
    faceid = MobileFacenet() #sphere20a(classnum=len(dataset_train.classes))
    faceid = faceid.cuda()

#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0001)
#     optimizer = optim.Adam(faceid.parameters(), lr=0.001)

#     criterion = nn.MSELoss().cuda()
#     denoise_criterion = nn.L1Loss().cuda()
    ArcMargin = ArcMarginProduct(128, len(dataset_train.classes))
    ArcMargin = ArcMargin.cuda()
    faceid_criterion = nn.CrossEntropyLoss().cuda()

    ignored_params = list(map(id, faceid.linear1.parameters()))
    ignored_params += list(map(id, ArcMargin.weight))
    prelu_params_id = []
    prelu_params = []
    for m in faceid.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params, faceid.parameters())
    lr_milstones = [36, 52, 58]
    lr = 0.1 
    optimizer = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': faceid.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': ArcMargin.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=lr, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, lr_milstones, gamma=0.1)

    models_dict = {
                'faceid': {
                     'model': faceid,
                },
                'arcmargin': {
                     'model': ArcMargin,
                }
              }
    schedulers_dict = {'general': scheduler}
    optimizers_dict = {'general': optimizer}
    losses_dict = {#'L1': denoise_criterion,
                   'FaceID': faceid_criterion}
    log_names = ["faceid_loss"] #'denoiser_loss', 
    
    trainer = JointTrainer(args.name, models_dict, schedulers_dict, optimizers_dict, losses_dict, log_names)
    trainer.train(dataloader_train, args.epochs)

    #for l1, l2 in zip(parameters_start,list(model.parameters())):
    #    print(np.array_equal(l1.data.numpy(), l2.data.numpy()))
    print("Done.")
