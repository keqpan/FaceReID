#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch.optim as optim
import numpy as np

from networks.faceid.sphereface import sphere20a

import torchvision
from torchvision import transforms

from loss import AngleLoss

from utils import printoneline, dt

  
ckpt_dir = "ckpt/"
logs_dir = "logs/"

def freeze_model(model):
#     model.train(False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def prepare_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model():
    ckpt_path = ckpt_dir + args.name + '_{}_{}.pth'.format("sphere20a", epoch)
    torch.save(model.state_dict(), ckpt_path)

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
    outputs = faceid(imgs)
    loss = faceid_criterion(outputs, labels)
    loss.backward()
    
#         torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 10)
#         torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 10)
    
    optimizer.step()

    cur_loss = loss.data.cpu().numpy().item()
    train_loss_arr.append(cur_loss)
    total_loss += cur_loss

    outputs = outputs[0] # 0=cos_theta 1=phi_theta
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += predicted.eq(labels.data).sum().cpu().item()

    grads = []
    for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
        grads.append([idx, p.grad.data.norm(2).item()])
    cur_grad_norm = np.sum(grads)

    printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | acc: %.4f%% faceid: %.4f grad: %.4f' % 
                 (epoch, total_loss/(batch_idx+1), batch_idx, 100. * correct/total, cur_loss, 
                 cur_grad_norm))

    return loss, total, correct, total_loss
        
def train_epoch(dataloader, optimizer, total, correct, total_loss, train_loss_arr):
    for batch_idx, sample in enumerate(dataloader):
        if stop_flag:
            break
        loss, total, correct, total_loss = global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr)

    return total, correct, total_loss

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
    
    train_data_dir = "/tmp/1st_udnet7pa_output"
    transform = transforms.Compose([
                             transforms.ToTensor()
                         ])
    dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
    dataloader_train = torch.utils.data.dataloader.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=14)
    
    faceid = sphere20a()
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
    faceid_ckpt_path = "/home/safin/ckpt/sphereface_clean/sphere20a_19.pth"
    faceid.load_state_dict(torch.load(faceid_ckpt_path))
    faceid = faceid.cuda()

#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0001)
#     optimizer = optim.Adam(faceid.parameters(), lr=0.001)

#     criterion = nn.MSELoss().cuda()
#     denoise_criterion = nn.L1Loss().cuda()
    faceid_criterion = AngleLoss().cuda()
    
    cur_logs_path = os.path.join(logs_dir, args.name)
    prepare_path(cur_logs_path)
    
    cur_ckpt_path = os.path.join(ckpt_dir, args.name)
    prepare_path(cur_ckpt_path)
    faceid_ckpt_path = os.path.join(cur_ckpt_path, "faceid")
    prepare_path(faceid_ckpt_path)
    
    total_train_loss_arr = []
    total_train_acc_arr = []

    lr_milstones = [5, 10, 20, 40]
#     scheduler = MultiStepLR(optimizer, lr_milstones, gamma=0.9)
    lr = 0.01 #
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
#         scheduler.step()
        if epoch in [0,10,15,18]:
            if epoch!=0: lr *= 0.5 #lr *= 0.9
            optimizer = optim.SGD(faceid.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        total, correct, total_loss = train_epoch(dataloader_train, optimizer, total, correct, total_loss, train_loss_arr)
    
        torch.save(faceid.state_dict(), os.path.join(faceid_ckpt_path, "weights_%d" % epoch))
#         torch.save(denoiser.state_dict(), os.path.join(denoiser_ckpt_path, "weights_%d" % epoch))
        
        total_train_loss_arr.append(np.mean(train_loss_arr))
        np.save(os.path.join(cur_logs_path,"train_loss_" + args.name), np.asarray(total_train_loss_arr))
        
        total_train_acc_arr.append(100. * correct/total)
        np.save(os.path.join(cur_logs_path,"train_faceid_acc_" + args.name), np.asarray(total_train_acc_arr))

        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
        np.save(os.path.join(cur_logs_path,"train_grads_" + args.name  + "_%d" % epoch), np.asarray(grads))
        print("\n")
        
#         total = 0
#         correct = 0
#         train_loss_arr = []
#         total_loss = 0
#         train_epoch(dataloader_val, None, total, correct, total_loss, train_loss_arr)
#         print("\n")
#         torch.save(denoiser.state_dict(), ckpt_path + "denoiser_" + args.name + "_%d" % epoch)
#         np.save("train_loss_" + args.name + "_%d" % epoch, np.asarray(train_loss_arr))

        
        if stop_flag:
            break
    #for l1, l2 in zip(parameters_start,list(model.parameters())):
    #    print(np.array_equal(l1.data.numpy(), l2.data.numpy()))
    print("Done.")