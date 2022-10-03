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



def get_imkey_ext(impath):
    imname=os.path.basename(impath)
    imkey=imname.split('.')[0]
    ext=imname.replace(imkey,'')
    return imkey,ext


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



def load_calva_land(txtpath):
    with open(txtpath,'r') as f:
        lines=f.readlines()
    faceland=str2pts(lines[0])
    portait_ext_land = str2pts(lines[1])
    etou_land = str2pts(lines[2])


    return  faceland,portait_ext_land,etou_land





if __name__=='__main__':
    srcroot=r'/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/ffhq'

    ims = get_ims(srcroot)

    for i, im in enumerate(ims):
        imkey, ext=get_imkey_ext(im)
        print(imkey, ext)
        txtpath=os.path.join(srcroot,imkey+'_calvaland.txt')

        if imkey!='FFHQ_01487':
            continue

        if os.path.exists(txtpath) is False:
            continue

        faceland,portait_ext_land,etou_land=load_calva_land(txtpath)
        img=cv2.imread(im)

        draw_pts(img, list(faceland), 10, (0, 255, 0), 5)
        # draw_pts(img, list(portait_ext_land), 10, (255, 255, 0), 5)
        draw_pts(img, list(etou_land), 20, (0, 255, 255), 5)

        cv2.imshow('img',limit_img_auto(img))

        if cv2.waitKey(0)==27:
            exit(0)










