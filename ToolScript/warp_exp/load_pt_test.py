# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import shutil
import platform,random
# import dlib,os
import numpy
# from skimage import io
import cv2
import numpy as np

import numpy as np
from delaunay2D import Delaunay2D

# Create a random set of points
seeds = np.random.random((10, 2))

# Create Delaunay Triangulation and insert points one by one


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

    ptlist=[1265, 2029 ,1379 ,1681, 1576 ,1398 ,1933 ,1302, 2258, 1413, 2453, 1695, 2567, 2038]
    ptsnp=np.array(ptlist).reshape(7,2)

    img=np.zeros((3840,3840,3),dtype=np.uint8)

    for pt in ptsnp:
        cv2.circle(img,tuple(pt), 20, (255,0,0), thickness=10, lineType=-1)

    dt = Delaunay2D()
    for pt in ptsnp:
        dt.addPoint(pt)
    trires=dt.exportTriangles()
    print(trires)
    for triind in trires:
        for k in range(0,3):
            cv2.line(img, tuple(ptsnp[triind[k % 3]]), tuple(ptsnp[triind[(k +1)% 3]]), (0, 255, 0), 10)




    cv2.imshow('img',limit_img_auto(img))
    key=cv2.waitKey(0)
    if key==27:
        exit(0)


    print('finish')