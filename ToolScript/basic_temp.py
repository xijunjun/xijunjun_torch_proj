# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import shutil
import platform,random
import numpy as np


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

if __name__ == '__main__':

    print('finish')