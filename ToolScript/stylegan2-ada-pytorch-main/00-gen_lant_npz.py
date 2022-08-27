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

def torch2np(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgnp = img.cpu().numpy()
    imgnp = np.squeeze(imgnp, axis=0)
    imgnp = cv2.cvtColor(imgnp, cv2.COLOR_BGR2RGB)
    return  imgnp

if __name__ == '__main__':

    baseroot='/disks/disk1/Workspace/TrainResult/stylegan/image_optimize/exp00'
    lantroot=os.path.join(baseroot,'lantcode')
    imageroot=os.path.join(baseroot,'image')
    makedir(lantroot)
    makedir(imageroot)

    network_pkl='/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq.pkl'
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    truncation_psi=1.0


    numseed=50
    for  i in range(0,numseed):


        seed_idx=i
        seed=random.randint(0,20000)

        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, numseed))
        seednpz=np.random.RandomState(seed).randn(1, G.z_dim)

        z = torch.from_numpy(seednpz).to(device)
        label = torch.zeros([1, G.c_dim], device=device)

        lantws=G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
        lantw_np=lantws.cpu().numpy()
        img=G.synthesis(lantws, noise_mode='const')
        # img2 = G.synthesis(lantws, noise_mode='const')


        # img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')

        # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        imgnp=torch2np(img)

        npkey=str(i).zfill(5)

        # print(imgnp.shape)
        # cv2.imshow('img',imgnp)
        # if cv2.waitKey(0)==27:
        #     exit(0)

        print(lantw_np.shape)
        np.save(os.path.join(lantroot,npkey+'.npy'),lantw_np)
        cv2.imwrite(os.path.join(imageroot,npkey+'.png'),imgnp)
        # cv2.imwrite(os.path.join(imageroot, npkey + '-b.png'), torch2np(img2))

    print('finish')