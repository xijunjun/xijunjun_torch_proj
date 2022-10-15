#coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform
import torch


def limit_img_auto(imgin):
    img = np.array(imgin)
    sw = 1920 * 1.2
    sh = 1080 * 1.2
    h, w = tuple(list(img.shape)[0:2])
    swhratio = 1.0 * sw / sh
    whratio = 1.0 * w / h
    # 横向的长图
    if whratio > swhratio:
        tw = int(sw)
        if tw < w:
            th = int(h * (tw / w))
            img = cv2.resize(img, (tw, th))
    else:
        th = int(sh)
        if th < h:
            tw = int(w * (th / h))
            img = cv2.resize(img, (tw, th))
    return img

def draw_pts(img,ptlist,r,color,thick,wait=0):
    for pt in ptlist:
        # print(pt)
        cv2.circle(img,tuple(np.array(pt,np.int32)),r,color,thick)
        if wait!=0:
            cv2.imshow('calva_mat_new', limit_img_auto(img))
            cv2.waitKey(wait)

def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst


def pts2str(pts):
    pts = list(np.array(pts, np.int32))
    resstr = ''
    for pt in pts:
        resstr += str(pt[0]) + ' ' + str(pt[1]) + ','
    resstr = resstr.rstrip(',')
    return resstr

def str2pts(line):
    line=line.rstrip('\n')
    items=line.split(',')
    pts=[]
    for item in items:
        corditems=item.split(' ')
        x=int(corditems[0])
        y=int(corditems[1])
        pts.append([x,y])
    return pts



def load_hair_rct(txtpath):

    with open(txtpath,'r') as f:
        lines=f.readlines()
    line=lines[0]
    items=line.split(',')
    ptstr1,ptstr2=items[0],items[1]
    pt1=np.fromstring(ptstr1,dtype=np.int32,sep=' ')
    pt2 = np.fromstring(ptstr2, dtype=np.int32, sep=' ')
    return  pt1,pt2



def get_imkey_ext(imname):
    imname=os.path.basename(imname)
    ext='.'+imname.split('.')[-1]
    imkey=imname.replace(ext,'')
    return imkey,ext

if __name__=='__main__':
    srcroot=r'/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/taobao_crop_good'

    ims = get_ims(srcroot)

    for i, im in enumerate(ims):
        imkey, ext=get_imkey_ext(im)
        print(imkey, ext)
        txtpath=os.path.join(srcroot,imkey+'.txt')

        # if imkey!='FFHQ_01487':
        #     continue

        if os.path.exists(txtpath) is False:
            continue

        pt1,pt2=load_hair_rct(txtpath)
        img=cv2.imread(im)

        cv2.rectangle(img, tuple(pt1), tuple(pt2), (255, 0, 0), 10, 1)

        cv2.imshow('img',limit_img_auto(img))

        if cv2.waitKey(0)==27:
            exit(0)










