#coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2,random
import os
import numpy as np
import shutil
import platform
import torch
import argparse
import cv2
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
import torchvision
from utils.seg_utils import save_vis_mask,vis_seg


def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def preprocess_img( img):
    im = torchvision.transforms.ToTensor()(img)[:3].unsqueeze(0).to('cuda:0')
    # im = (downsample(im).clamp(0, 1) - seg_mean) / seg_std
    im = (im.clamp(0, 1) - seg_mean) / seg_std
    return im

if __name__=='__main__':


    device='cuda:0'
    seg = BiSeNet(n_classes=16)
    seg.to(device)
    seg_ckpt='/home/tao/Downloads/seg.pth'

    seg.load_state_dict(torch.load(seg_ckpt))
    for param in seg.parameters():
        param.requires_grad = False
    seg.eval()


    imroot='/disks/disk1/Workspace/Project/Pytorch/FaceEdit/Barbershop-main/input/face'
    ims=get_ims(imroot)
    for im in ims:
        img=cv2.imread(im)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=preprocess_img(img)

        down_seg, _, _ = seg(img)
        gen_seg_target = torch.argmax(down_seg, dim=1).long()
        gen_seg_target=gen_seg_target[0].cpu()
        print(gen_seg_target.shape)

        vis_mask = vis_seg(gen_seg_target)

        # PIL.Image.fromarray(vis_mask).save(vis_path)

        cv2.imshow('img',vis_mask)
        key=cv2.waitKey(0)
        if key==27:
            exit(0)






