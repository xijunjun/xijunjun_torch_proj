

#coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform

from multiprocessing import Process
import time,numpy as np
import os,cv2

import cv2
import torch
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter


import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()
def to_pil_image(input):
    return np.array(transforms.ToPILImage()(input))


def makedir(dirtp):
    if os.path.exists(dirtp):
        return
    os.makedirs(dirtp)

def quad_dim1to2(quad):
    return np.reshape(np.array(quad),(4,2))

def limit_img_auto(img):
    sw=1920*1.0
    sh=1080*1.0
    h,w,c=img.shape
    swhratio=1.0*sw/sh
    whratio=1.0*w/h
    if whratio>swhratio:
        th=int(sh)
        if th>h:
            return img
        tw=int(w*(th/h))
        img=cv2.resize(img,(tw,th))
    else:
        tw=int(sw)
        if tw>w:
            return img
        th=int(h*(tw/w))
        img=cv2.resize(img,(tw,th))
    return  img

def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def cvt_tensor_color(imtensor_):
    imtensor=torch.tensor(imtensor_)
    temp=torch.tensor(imtensor[0])
    imtensor[0]=imtensor[2]
    imtensor[2]=temp

    print(temp.shape)
    return  imtensor


if __name__=='__main__':

    srcroot='/disks/disk1/Dataset/Project/SuperResolution/taobao_stand_face'
    ims=get_ims(srcroot)

    writer = SummaryWriter(log_dir='tsbd')
    for i,im in enumerate(ims):
        img=cv2.imread(im)
        img_tensor = to_tensor(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        img_from_tensor = to_pil_image(img_tensor[0])

        print(img.shape,img_tensor.shape,img_from_tensor.shape)




        cv2.imshow('img',img)
        cv2.imshow('img_from_tensor',img_from_tensor)

        writer.add_image('input', cvt_tensor_color(img_tensor[0]), i)
        # writer.add_image('input', img_tensor[0], i)

        # writer.add_image('numpy', img, i)

        cv2.imshow('11',to_pil_image(cvt_tensor_color(img_tensor[0])))


        if cv2.waitKey(0)==27:
            writer.close()
            exit(0)



    print('finish')




