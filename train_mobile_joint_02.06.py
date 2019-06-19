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

from utils import printoneline, dt, load_model
from transforms.noising import RawNoise
from datasets.noised import DemosaicDataset
from networks.faceid.mobile import MobileFacenet
from networks.faceid.mobile import ArcMarginProduct
from loss import PSNR

from networks.denoise.pydl import ResNet_Den

from MMNet_TBPTT import *
from problems import *
import torch.nn.functional as F

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler

from common import CKPT_DIR, LOGS_DIR
import itertools
from collections import OrderedDict

def freeze_model(model):
#     model.train(False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def prepare_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_mask(im_shape, pattern='RGGB'):
    if pattern == 'RGGB':
        # pattern RGGB
        r_mask = torch.zeros(im_shape)
        r_mask[0::2, 0::2] = 1

        g_mask = torch.zeros(im_shape)
        g_mask[::2, 1::2] = 1
        g_mask[1::2, ::2] = 1

        b_mask = torch.zeros(im_shape)
        b_mask[1::2, 1::2] = 1

        mask = torch.zeros(im_shape + (3,))
        mask[:, :, 0] = r_mask
        mask[:, :, 1] = g_mask
        mask[:, :, 2] = b_mask
        
        return mask
       
from base import BaseExpRunner

class JointTrainer(BaseExpRunner):
    num_avg_batches = 32 # 64 # 32
    T = 0.1
    alpha = 0.9
    
    def global_forward(self, sample, batch_idx):
        mosaic, groundtruth, labels = sample
        mosaic, groundtruth, labels = mosaic.cuda(non_blocking=True), groundtruth.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        if (batch_idx+1)%self.num_avg_batches == 0:
            make_step = True
        else:
            make_step = False

        if batch_idx%self.num_avg_batches == 0:
            zero_grad = True
        else:
            zero_grad = False
            
            
        denoised_imgs = mmnet.forward_all_iter(mosaic, M, init=False, noise_estimation=True, max_iter=None)
        denoiser_loss = denoise_criterion(denoised_imgs, groundtruth)
        denoiser_loss = denoiser_loss
        faceid_input = 255*linrgb_to_srgb(denoised_imgs/255)/0.6
        faceid_input = (faceid_input-127.5)/128
        raw_logits = faceid(faceid_input)
        outputs = arcmargin(raw_logits, labels)
        
        clean_faceid_input = 255*linrgb_to_srgb(groundtruth/255)/0.6
        clean_faceid_input = (clean_faceid_input-127.5)/128
        raw_logits_t = faceid_t(clean_faceid_input)
        teacher_outputs = arcmargin_t(raw_logits_t, labels)
        faceid_loss = faceid_criterion(outputs, labels)
#         print(F.log_softmax(outputs/self.T, dim=1))
#         print("student:", F.softmax(outputs/self.T, dim=1))
#         print("teacher:", F.softmax(teacher_outputs/self.T, dim=1))
        kd_loss_part = 1000*nn.KLDivLoss()(F.log_softmax(outputs/self.T, dim=1),
                             F.softmax(teacher_outputs/self.T, dim=1)) 
#         kd_loss_part = nn.MSELoss()(outputs, teacher_outputs)
        kd_loss = kd_loss_part * (self.alpha * self.T * self.T)
        faceid_loss = (1. - self.alpha) * faceid_loss

        loss = (denoiser_loss+faceid_loss+kd_loss)/self.num_avg_batches
        denoiser_loss = float(denoiser_loss)
        faceid_loss = float(faceid_loss)
        kd_loss = float(kd_loss)
        
        if zero_grad:
            mmnet.zero_grad()
            faceid.zero_grad()
            arcmargin.zero_grad()
        # backprop last module (keep graph only if they ever overlap)
        loss.backward(retain_graph=False)
        
        grads = []
        for idx, par in enumerate(list(filter(lambda p: p.grad is not None, mmnet.parameters()))):
            grads.append(par.grad.data.norm(2).item())
        cur_grad_dn_norm = np.sum(grads)

        grads = []
        for idx, par in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
            grads.append(par.grad.data.norm(2).item())
        cur_grad_faceid_norm = np.sum(grads)
        
        grads = []
        for idx, par in enumerate(list(filter(lambda p: p.grad is not None, arcmargin.parameters()))):
            grads.append(par.grad.data.norm(2).item())
        cur_grad_arcmargin_norm = np.sum(grads)
        
#         torch.nn.utils.clip_grad_norm_(mmnet.parameters(), 0.25)
        if make_step:
            optimizer.step()

#         denoised_imgs, cur_grad_dn_norm, cur_grad_faceid_norm, cur_grad_arcmargin_norm, denoiser_loss, faceid_loss = runner.train(mosaic, M, groundtruth, labels, init=False, noise_estimation=True, zero_grad=zero_grad, make_step=make_step, num_avg_batches=self.num_avg_batches)

        cur_psnr = float(PSNR(linrgb_to_srgb(denoised_imgs/255)/0.6, linrgb_to_srgb(groundtruth/255)/0.6).mean())
    #     psnr_list = calculate_psnr_fast(denoised_imgs/255, groundtruth/255)
    #     mean_psnr = np.array(psnr_list)
    #     mean_psnr = mean_psnr[mean_psnr != np.inf].mean()
        cur_loss = denoiser_loss + faceid_loss + kd_loss_part
        self.tmp_logs_dict['denoiser_loss'].append(denoiser_loss)
        self.tmp_logs_dict['faceid_loss'].append(faceid_loss)
        self.tmp_logs_dict['kd_loss'].append(kd_loss)

        self.total_loss += cur_loss # FIXME
        if batch_idx % 50 == 0:
            printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | denoise: %.4f faceid: %.4f kd: %.4f| gradDN: %.4f gradID: %.4f gradM: %.4f psnr: %.4f' % 
                     (self.cur_epoch, self.total_loss/(batch_idx+1), batch_idx, denoiser_loss, faceid_loss, float(kd_loss_part), cur_grad_dn_norm, cur_grad_faceid_norm, cur_grad_arcmargin_norm, cur_psnr))


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
    else:
        multigpu_mode = False
    
    M = generate_mask((112, 96), pattern='RGGB').permute(2,0,1).unsqueeze(0).cuda()
    if multigpu_mode:
        M = M.repeat(args.batch_size,1,1,1)

    
    train_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_train_idxs.npy")
    val_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_test_idxs.npy")
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices[:50])

#     a = 0.15
#     b = 0.15
#     dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)

    train_data_dir = "/home/safin/datasets/CASIA-WebFace_linRGB/"
    transform = transforms.Compose([
                     transforms.CenterCrop((112,96)),
                     transforms.ToTensor()
                ])
    noised_dataset = DemosaicDataset(train_data_dir, transform)

    dataloader_train = torch.utils.data.dataloader.DataLoader(noised_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=32)
    dataloader_val = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=val_sampler, batch_size=8, pin_memory=True, num_workers=12)
    
    denoiser = ResNet_Den(5, weightnorm=True)
    
    max_iter = 5
    mmnet = MMNet(denoiser, max_iter=max_iter)
    
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobilefacenet_08.05/faceid/weights_70"
    faceid_t = MobileFacenet()
    faceid_t = load_model(faceid_t, faceid_ckpt_path, multigpu_mode=False, use_cuda=True)
    arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/mobilefacenet_08.05/arcmargin/weights_70"
    arcmargin_t = ArcMarginProduct(128, len(noised_dataset.classes))
    arcmargin_t = load_model(arcmargin_t, arcmargin_ckpt_path, multigpu_mode=False, use_cuda=True)
    
#     freeze_model(mmnet)
    
    faceid = MobileFacenet() 
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_dnfr_16.04/faceid/weights_30"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobilefacenet_08.05/faceid/weights_70"

#     faceid_state_dict_gpu = torch.load(faceid_ckpt_path, map_location=lambda storage, loc: storage)
#     faceid_state_dict = OrderedDict()
#     for k, v in faceid_state_dict_gpu.items():
#         faceid_state_dict[k[7:]] = v
#     faceid.load_state_dict(faceid_state_dict)
#     faceid = load_model(faceid, faceid_ckpt_path, multigpu_mode)
    arcmargin = ArcMarginProduct(128, len(noised_dataset.classes))

    
    denoise_criterion = nn.L1Loss().cuda()
    # denoise_criterion = nn.MSELoss().cuda()
    faceid_criterion = nn.CrossEntropyLoss().cuda()


    optimized_params = itertools.chain(mmnet.parameters(), faceid.parameters(), arcmargin.parameters())
#     optimized_params = itertools.chain(faceid.parameters(), arcmargin.parameters())
#     optimized_params = mmnet.parameters()
    
    ignored_params = list(map(id, faceid.linear1.parameters()))
    ignored_params += list(map(id, arcmargin.weight))
    prelu_params_id = []
    prelu_params = []
    for m in faceid.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params, faceid.parameters())
    lr_milstones = [2, 5]
    lr = 0.0005
    optimizer = torch.optim.Adam(optimized_params, lr=lr, amsgrad=True)
#     optimizer = optim.SGD([
#         {'params': base_params, 'weight_decay': 4e-5},
#         {'params': faceid.linear1.parameters(), 'weight_decay': 4e-4},
#         {'params': arcmargin.weight, 'weight_decay': 4e-4},
#         {'params': prelu_params, 'weight_decay': 0.0}
#     ], lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milstones, gamma=0.1)

    runner = TBPTT_faceid(mmnet, faceid, denoise_criterion, faceid_criterion, arcmargin, 5, 5, optimizer, max_iter=max_iter, clip_grad=0.25)
    runner = runner.cuda()

#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_13.05/faceid/weights_6"
#     arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/joint_13.05/arcmargin/weights_6"
    mmnet_ckpt_path = "/home/safin/FaceReID/ckpt/mmnet_5it_26.05_3/mmnet/weights_299"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mmnet_by_l1+faceid/faceid/weights_14"
#     arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/mmnet_by_l1+faceid/arcmargin/weights_14"
    
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_on_mnnet_27.05/faceid/weights_16"
#     arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_on_mnnet_27.05/arcmargin/weights_16"
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobilefacenet_08.05/faceid/weights_70"
    arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/mobilefacenet_08.05/arcmargin/weights_70"
    
    models_dict = {
                    'mmnet': {
                        'model': mmnet,
                        'load_ckpt': mmnet_ckpt_path
                    },
                    'faceid': {
                         'model': faceid,
                         'load_ckpt': faceid_ckpt_path #"/home/safin/FaceReID/ckpt/mobile_16.04/faceid/weights_60" 
#                         "/home/safin/FaceReID/ckpt/joint_dnfr_22.04/faceid/weights_23"
                    },
                    'arcmargin': {
                         'model': arcmargin,
                         'load_ckpt': arcmargin_ckpt_path #"/home/safin/FaceReID/ckpt/mobile_16.04/arcmargin/weights_60" 
#                         "/home/safin/FaceReID/ckpt/joint_dnfr_22.04/argcmargin/weights_23"
                    }
                  }
#     freeze_model(faceid)
#     freeze_model(arcmargin)
    schedulers_dict = {'general': scheduler}
    optimizers_dict = {'general': optimizer}
    losses_dict = {'L1': denoise_criterion,
                   'FaceID': faceid_criterion}
    log_names = ['denoiser_loss', "faceid_loss", "kd_loss"]
    
    trainer = JointTrainer(args.name, models_dict, schedulers_dict, optimizers_dict, losses_dict, log_names)
    trainer.train(dataloader_train, args.epochs)

#         grads = []
#         for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
#             grads.append([idx, p.grad.data.norm(2).item()])
#         np.save(os.path.join(cur_logs_path,"train_grads_" + args.name  + "_%d" % epoch), np.asarray(grads))

    print("Done.")
