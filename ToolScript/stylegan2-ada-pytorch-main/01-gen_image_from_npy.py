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

if __name__ == '__main__':

    baseroot='/disks/disk1/Workspace/TrainResult/stylegan/image_optimize/exp00'
    lantroot=os.path.join(baseroot,'lantcode')
    imageroot=os.path.join(baseroot,'image_re')
    makedir(imageroot)

    network_pkl='/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq.pkl'
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    truncation_psi=1.0

    npys=get_npys(lantroot)

    numseed=50
    for  i,npypath in enumerate(npys):


        seed_idx=i
        lantws_np=np.load(npypath)
        npkey=os.path.basename(npypath).split('.')[0]


        print('Generating image for seed (%d/%d) ...' % ( seed_idx, numseed))
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
        cv2.imwrite(os.path.join(imageroot,npkey+'.png'),imgnp)

    print('finish')