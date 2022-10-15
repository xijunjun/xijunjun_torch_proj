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



global global_ptsel
global_ptsel=0
global global_hairrct
global_hairrct=[None,None]


def refresh_vis():

    global global_ptsel
    global global_hairrct,visimg

    visimgshow=np.array(visimg)
    pt1=global_hairrct[0]
    pt2=global_hairrct[1]
    cv2.rectangle(visimgshow, tuple(pt1), tuple(pt2), (255, 0, 0), 10, 1)

    cv2.circle(visimgshow, tuple(global_hairrct[global_ptsel]), 20, (0, 0, 255), 10, -1)

    cv2.imshow('img',visimgshow)


def euclidean(pt1,pt2):
    return  np.linalg.norm(np.array(pt1) - np.array(pt2))

def min_dis_ind(pt1,ptlist):
    dis_list=[]
    for pt in ptlist:
        dis_list.append(euclidean(pt,pt1))
    dis_list=np.array(dis_list)
    ind=np.argmin(dis_list)
    return  ind


def add_pts(event,x,y,flags,param):
    global global_hairrct,visimg,global_ptsel
    h,w,c=visimg.shape

    if event == cv2.EVENT_LBUTTONDOWN:
        if  global_ptsel==0:
            global_hairrct[global_ptsel]=[x,y]
        if  global_ptsel==1:
            global_hairrct[global_ptsel]=[x,h-1]

        refresh_vis()
    if event==cv2.EVENT_MOUSEMOVE:
        global_ptsel=min_dis_ind([x,y],global_hairrct)
        refresh_vis()


def makedir(dirtp):
    if os.path.exists(dirtp):
        return
    os.makedirs(dirtp)

global visimg
if __name__=='__main__':
    srcroot=r'/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/taobao_crop'

    dstroot_good=srcroot+'_good'
    dstroot_bad=srcroot+'_bad'

    makedir(dstroot_bad)
    makedir(dstroot_good)

    ims = get_ims(srcroot)


    scale_factor=2.0

    for i, im in enumerate(ims):

        imname=os.path.basename(im)
        imkey, ext=get_imkey_ext(imname)
        print(imkey, ext)
        txtpath=os.path.join(srcroot,imkey+'.txt')
        txtname=imkey+'.txt'


        # if imkey!='FFHQ_01487':
        #     continue
        img = cv2.imread(im)
        img=cv2.resize(img,None,fx=scale_factor,fy=scale_factor)

        visimg=np.array(img)
        h,w,c=img.shape

        pt1=[0,0]
        pt2=[w,h]

        if os.path.exists(txtpath):
            pt1,pt2=load_hair_rct(txtpath)

        global_hairrct[0]=(pt1*scale_factor).astype(np.int32)
        global_hairrct[1]=(pt2*scale_factor).astype(np.int32)

        refresh_vis()
        cv2.setMouseCallback('img', add_pts)

        while 1:
            key=cv2.waitKey(0)
            print(key)
            if key==13:
                impath_src=os.path.join(srcroot,imname)
                impath_dst=os.path.join(dstroot_good,imname)
                txtpath_dst=os.path.join(dstroot_good,txtname)

                if os.path.exists(txtpath):
                    shutil.move(txtpath,txtpath_dst)
                shutil.move(impath_src,impath_dst)

                pt_tl=np.array(np.array(global_hairrct[0])/scale_factor,np.int32)
                pt_br = np.array(np.array(global_hairrct[1]) / scale_factor, np.int32)

                anolines = str(pt_tl[0]) + ' ' + str(pt_tl[1]) + ',' + str(pt_br[0]) + ' ' + str(pt_br[1])

                with open(txtpath_dst,'w') as f:
                    f.writelines(anolines)
                break
            if key==8:
                impath_src=os.path.join(srcroot,imname)
                impath_dst=os.path.join(dstroot_bad,imname)
                txtpath_dst=os.path.join(dstroot_bad,txtname)

                if os.path.exists(txtpath):
                    shutil.move(txtpath,txtpath_dst)
                shutil.move(impath_src,impath_dst)

                pt_tl=np.array(np.array(global_hairrct[0])/scale_factor,np.int32)
                pt_br = np.array(np.array(global_hairrct[1]) / scale_factor, np.int32)

                anolines = str(pt_tl[0]) + ' ' + str(pt_tl[1]) + ',' + str(pt_br[0]) + ' ' + str(pt_br[1])

                with open(txtpath_dst,'w') as f:
                    f.writelines(anolines)
                break






            if cv2.waitKey(0)==27:
                exit(0)










