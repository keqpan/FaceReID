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

stop_flag = False
def handler(signum, frame):
    print("Shutting down at " + dt() + " ...")
    global stop_flag
    stop_flag = True

import signal
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

def global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr):
    optimizer.zero_grad()
    imgs, labels = sample
    imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

    imgs = (imgs*255 - 127.5)/128
    # compute output
    optimizer.zero_grad()
    raw_logits = faceid(imgs)
    outputs = ArcMargin(raw_logits, labels)
    loss = faceid_criterion(outputs, labels)
    loss.backward()
    
#         torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 10)
#         torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 10)
    
    optimizer.step()

    cur_loss = float(loss)
    train_loss_arr.append(cur_loss)
    total_loss += cur_loss
#     print(outputs.shape)
#     outputs = outputs[0] # 0=cos_theta 1=phi_theta
    _, predicted = torch.max(raw_logits.data, 1)
    total += labels.size(0)
    correct += int(predicted.eq(labels.data).sum())

    grads = []
    for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
        grads.append([idx, p.grad.data.norm(2).item()])
    cur_grad_norm = np.sum(grads)
    
    grads = []
    for idx, p in enumerate(list(filter(lambda p: p.grad is not None, ArcMargin.parameters()))):
        grads.append([idx, p.grad.data.norm(2).item()])
    arcmargin_grad_norm = np.sum(grads)
    
    if batch_idx % 50 == 0:
        printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | acc: %.4f%% faceid: %.4f grad: %.4f gradM: %.4f' % 
                     (epoch, total_loss/(batch_idx+1), batch_idx, 100. * correct/total, cur_loss, 
                     cur_grad_norm, arcmargin_grad_norm))

    return loss, total, correct, total_loss
        
def train_epoch(dataloader, optimizer, total, correct, total_loss, train_loss_arr):
    for batch_idx, sample in enumerate(dataloader):
        if stop_flag:
            break
        loss, total, correct, total_loss = global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr)

    return total, correct, total_loss

sig_read_linspace = np.linspace(-3,-1.5,4)
sig_shot_linspace = np.linspace(-2,-1,4)

sig_read = sig_read_linspace[3]
sig_shot = sig_shot_linspace[2]
a = np.power(10., sig_read)
b = np.power(10., sig_shot)

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
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/01.01_sphere20a_20.pth" #trained for high noise
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/faceid_joint_3stages_20.01_8"
#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "ckpt/1stage_udnet_fixed_sphereface_27.01/faceid/faceid_1stage_udnet_fixed_sphereface_27.01_30"

#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_07.02_fixed/faceid/weights_74"
#     faceid_ckpt_path = "ckpt/1stage_udnet_sphereface_08.02_finetune/faceid/weights_"+str(n_ckpt)
#     faceid_ckpt_path = "ckpt/joint_07.02_finetune_08.02/faceid/weights_"+str(n_ckpt)
#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/sphereface_13.02/faceid/weights_10"
#     faceid_ckpt_path = "/home/safin/ckpt/sphereface_clean/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/sr_faceid_27.03/faceid/weights_13"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/sr_qmul_29.03/faceid/weights_12"
#     model_state = torch.load(faceid_ckpt_path)
#     del model_state['fc6.weight']
#     faceid.load_state_dict(model_state, strict=False)

    faceid = faceid.cuda()

#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0001)
#     optimizer = optim.Adam(faceid.parameters(), lr=0.001)

#     criterion = nn.MSELoss().cuda()
#     denoise_criterion = nn.L1Loss().cuda()
    ArcMargin = ArcMarginProduct(128, len(dataset_train.classes))
    ArcMargin = ArcMargin.cuda()
    faceid_criterion = nn.CrossEntropyLoss().cuda()

    
    cur_logs_path = os.path.join(LOGS_DIR, args.name)
    os.makedirs(cur_logs_path, exist_ok=True)
    
    cur_ckpt_path = os.path.join(CKPT_DIR, args.name)
    os.makedirs(cur_ckpt_path, exist_ok=True)
    faceid_ckpt_path = os.path.join(cur_ckpt_path, "faceid")
    os.makedirs(faceid_ckpt_path, exist_ok=True)
    arcmargin_ckpt_path = os.path.join(cur_ckpt_path, "arcmargin")
    os.makedirs(arcmargin_ckpt_path, exist_ok=True)
    
    total_train_loss_arr = []
    total_train_acc_arr = []

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

    if multigpu_mode:
        faceid = nn.DataParallel(faceid)
    if multigpu_mode:
        ArcMargin = nn.DataParallel(ArcMargin)
    
    for epoch in range(args.epochs):
        total = 0
        correct = 0
        train_loss_arr = []
        total_loss = 0
#         faceid_w = 1
#         denoise_w = 0
#         if epoch >= 1:
#             faceid_w = 1
#         else:
#             faceid_w = 0

        total, correct, total_loss = train_epoch(dataloader_train, optimizer, total, correct, total_loss, train_loss_arr)
    
        save_model(faceid, os.path.join(faceid_ckpt_path, "weights_%d" % epoch), multigpu_mode)
        save_model(ArcMargin, os.path.join(arcmargin_ckpt_path, "weights_%d" % epoch), multigpu_mode)
        
        total_train_loss_arr.append(np.mean(train_loss_arr))
        np.save(os.path.join(cur_logs_path,"train_loss_" + args.name), np.asarray(total_train_loss_arr))


#         grads = []
#         for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
#             grads.append([idx, p.grad.data.norm(2).item()])
#         np.save(os.path.join(cur_logs_path,"train_grads_" + args.name  + "_%d" % epoch), np.asarray(grads))
        print("\n")
        
        scheduler.step()
        if stop_flag:
            break
    #for l1, l2 in zip(parameters_start,list(model.parameters())):
    #    print(np.array_equal(l1.data.numpy(), l2.data.numpy()))
    print("Done.")
