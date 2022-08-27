# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import shutil
import platform,random
import numpy as np
import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import copy
import io
import torch.nn.functional as F

import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()
def to_np_image(input):
    return np.array(transforms.ToPILImage()(input))

def makedir(dirtp):
    if os.path.exists(dirtp):
        return
    os.makedirs(dirtp)

def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def get_npys(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.npy']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def limit_img_auto(imgin):
    img=np.array(imgin)
    sw=1920*1.0
    sh=1080*1.0
    h,w,c=img.shape
    swhratio=1.0*sw/sh
    whratio=1.0*w/h
    # 横向的长图
    if whratio>swhratio:
        tw=int(sw)
        if tw<w:
            th=int(h*(tw/w))
            img=cv2.resize(img,(tw,th))
    else:
        th=int(sh)
        if th<h:
            tw=int(w*(th/h))
            img=cv2.resize(img,(tw,th))
    return  img


def torch2np(imgin):
    # img=torch.tensor(imgin).requires_grad_(False)
    img = (imgin.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgnp = img.cpu().numpy()
    imgnp = np.squeeze(imgnp, axis=0)
    imgnp = cv2.cvtColor(imgnp, cv2.COLOR_BGR2RGB)
    return  imgnp


if __name__ == '__main__':

    # baseroot='/disks/disk1/Workspace/TrainResult/stylegan/image_optimize/exp00'
    # lantroot=os.path.join(baseroot,'lantcode')
    # imageroot=os.path.join(baseroot,'image_re')
    # makedir(imageroot)

    image_src_path='/disks/disk1/Workspace/TrainResult/stylegan/image_optimize/opt-test00/00009.png'
    image_dst_path='/disks/disk1/Workspace/TrainResult/stylegan/image_optimize/opt-test00/00009_dst.png'
    lantcode_path='/disks/disk1/Workspace/TrainResult/stylegan/image_optimize/opt-test00/00009.npy'

    network_pkl='/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq.pkl'
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    truncation_psi=1.0
    losssize=128



    npypath=lantcode_path
    lantws_np=np.load(npypath)
    npkey=os.path.basename(npypath).split('.')[0]

    lantws = torch.from_numpy(lantws_np).to(device)
    image_src=cv2.imread(image_src_path)
    image_dst = cv2.imread(image_dst_path)

    image_dst_rgb = cv2.cvtColor(image_dst, cv2.COLOR_BGR2RGB)
    target_tensor = torch.tensor(image_dst_rgb.transpose([2, 0, 1]), device=device)
    target_images = target_tensor.unsqueeze(0).to(device).to(torch.float32)
    target_images = F.interpolate(target_images, size=(losssize, losssize), mode='area')



    initial_learning_rate=0.01
    num_steps=1000
    # ############################
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    # Load VGG16 feature detector.
    vggpath='/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/pretrained/metrics/vgg16.pt'
    with open(vggpath,"rb") as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(lantws, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    # w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    # Setup noise inputs.
    # noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
    optimizer = torch.optim.Adam([w_opt] , betas=(0.9, 0.999), lr=initial_learning_rate)

    for step in range(num_steps):
        synth_images = G.synthesis(w_opt, noise_mode='const')
        # synth_images = G.synthesis(lantws, noise_mode='const')


        cur_image = torch2np(synth_images)
        synth_images = (synth_images + 1) * (255 / 2)
        # cur_image=to_np_image(synth_images[0])

        # lantws=torch.from_numpy(lantws_np).to(device)
        # img=G.synthesis(lantws, noise_mode='const')


        synth_images = F.interpolate(synth_images, size=(losssize, losssize), mode='area')
        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        loss = dist
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        print('loss:',loss)
        # Save projected W for each optimization step.
        # w_out[step] = w_opt.detach()[0]

        cv2.imshow('image_src',limit_img_auto(np.concatenate([image_src,cur_image,image_dst],axis=1)))
        if cv2.waitKey(30)==27:
            exit(0)
    exit(0)




    # seednpz=np.random.RandomState(seed).randn(1, G.z_dim)

    lantws=torch.from_numpy(lantws_np).to(device)
    # print(lantws_np.shape)
    print('---',lantws_np.shape,lantws.shape)

    img=G.synthesis(lantws, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    imgnp=img.cpu().numpy()
    imgnp=np.squeeze(imgnp,axis=0)
    imgnp=cv2.cvtColor(imgnp,cv2.COLOR_BGR2RGB)

    # npkey=str(i).zfill(5)

    # print(imgnp.shape)
    # cv2.imshow('img',imgnp)
    # if cv2.waitKey(0)==27:
    #     exit(0)

    # np.save(os.path.join(lantroot,npkey+'.npy'),seednpz)
    # cv2.imwrite(os.path.join(imageroot,npkey+'.png'),imgnp)

    print('finish')