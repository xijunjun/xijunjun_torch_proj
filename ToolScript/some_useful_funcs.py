
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


import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()
def to_np_image(input):
    return np.array(transforms.ToPILImage()(input))


# for i in range(0, 4):
#     cv2.line(disimg, tuple(oneplate[i % 4]), tuple(oneplate[(i + 1) % 4]), (0, 255, 0), int(2 / global_var[0]))



def image_1to3c(imagein):
    image3c=np.array(imagein[:,:,None])
    image3c=image3c.repeat(3,axis=2)
    return  image3c


def makedir(dirtp):
    if os.path.exists(dirtp):
        return
    os.makedirs(dirtp)

def quad_dim1to2(quad):
    return np.reshape(np.array(quad),(4,2))

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

def file_extension(path):
  return os.path.splitext(path)[1]

def read_list_lines(txtpathlist):
    lines = []
    for txtpath in txtpathlist:
        with open(txtpath, 'r') as f:
            tplines = f.readlines()
            for line in tplines:
                lines.append(line)
    return lines

def pts2rct(box):
    tlx = min(box[:,0])
    tly = min(box[:,1])
    brx = max(box[:,0])
    bry = max(box[:,1])
    return [tlx,tly,brx,bry]

def small_rct(rct, wratio, hratio):
  rctnew = np.array(rct, dtype=np.float32)
  tlx, tly = rct[0], rct[1]
  rct[::2] -= tlx
  rct[1::2] -= tly

  rctnew[::2] = rct[::2] * wratio
  rctnew[1::2] = rct[1::2] * hratio
  offw = ((rctnew[2] - rctnew[0]) - (rct[2] - rct[0])) // 2
  rctnew[::2] -= offw
  offh = ((rctnew[3] - rctnew[1]) - (rct[3] - rct[1])) // 2
  rctnew[1::2] -= offh
  rctnew[::2] += tlx
  rctnew[1::2] += tly

  return rctnew

def bigger_quad(quad,wratio,hratio):
    rct=np.array(pts2rct(quad_dim1to2(quad)),dtype=np.float32)
    tlx,tly=rct[0],rct[1]
    quad[::2]-=tlx
    quad[1::2] -= tly
    rctnew=np.array(rct,dtype=np.float32)
    quadnew=np.array(quad,dtype=np.float32)
    rctnew[::2]=rct[::2]*wratio
    rctnew[1::2] = rct[1::2] * hratio
    quadnew[::2]=quad[::2]*wratio
    quadnew[1::2] = quad[1::2] * hratio

    offw=((rctnew[2]-rctnew[0])-(rct[2]-rct[0]))//2
    offh=((rctnew[3]-rctnew[1])-(rct[3]-rct[1]))//2

    quadnew[::2] -= offw
    quadnew[1::2]-=offh
    quad[::2]+=tlx
    quad[1::2] += tly

    quadnew[::2]+=tlx
    quadnew[1::2] += tly
    quadnew=quadnew.astype(np.int)
    return quadnew

def bigger_rct(wratio,hratio,rct):
    wratio=(wratio-1.0)*0.5
    hratio=(hratio-1.0)*0.5
    delta_w=(rct[2])*wratio+0.5
    delta_h=(rct[3])*hratio+0.5
    # return limit_rct([int(rct[0]-delta_w),int(rct[1]-delta_h),int(rct[2]+delta_w*2),int(rct[3]+delta_h*2)],imgshape)
    return [int(rct[0]-delta_w),int(rct[1]-delta_h),int(rct[2]+delta_w*2),int(rct[3]+delta_h*2)]

def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst


def getpath(rootlist,imname):
    for root in rootlist:
        if os.path.exists(os.path.join(root,imname)):
            return os.path.join(root,imname)
    return None


def sum_img_hori(imglist,inter):
    rows=0;cols=0
    for img in imglist:
        cols+=img.shape[1]+inter
        if img.shape[0]>rows:
            rows=img.shape[0]
    sumimg = np.zeros((rows, cols-inter,3), imglist[0].dtype)
    xstart = 0
    for img in imglist:
        sumimg[0:img.shape[0],xstart:xstart+img.shape[1]]=img
        xstart=xstart+img.shape[1]+inter
    return sumimg

def sum_img_vertical(imglist):
    rows=0;cols=0
    for img in imglist:
        rows+=img.shape[0]+10
        if img.shape[1]>cols:
            cols=img.shape[1]
    sumimg = np.zeros((rows, cols,3), imglist[0].dtype)
    # sumimg=np.array(Image.new("RGB", (cols, rows), (0,0,0)))
    ystart = 0
    for img in imglist:
        sumimg[ystart:ystart+img.shape[0],0:img.shape[1]]=img
        ystart=ystart+img.shape[0]+10
    return sumimg

def draw_charimdict(imglist):
    numim=len(imglist)
    colnum=10
    rownum=numim//colnum
    if rownum*colnum<numim:
        rownum+=1
    h,w,c=imglist[0].shape
    sumw=w*colnum
    sumh=h*rownum
    sumimg=np.zeros((sumh,sumw,3),imglist[0].dtype)+255
    for i in range(0,rownum):
        for j in range(0,colnum):
            imind=colnum*i+j
            if imind>=numim:
                break
            sumimg[i*h:i*h+h,j*w:j*w+w]=imglist[colnum*i+j]
    return  sumimg

def get_imkey(impath):
    return os.path.basename(impath).replace('.'+impath.split('.')[-1],'')



def crop_pad(img,bdrct):
    h,w,c=img.shape

    extend_tlx = min(0,bdrct[0][0])
    extend_tly = min(0, bdrct[0][1])
    extend_brx = max(w,bdrct[1][0])
    extend_bry = max(h, bdrct[1][1])
    extendw=extend_brx-extend_tlx
    extendh = extend_bry - extend_tly
    bkimg=np.zeros((extendh,extendw,3),img.dtype)
    # print('cropimg',extend_tlx,extend_tly)

    xshift=0-extend_tlx
    yshift = 0 - extend_tly
    bkimg[yshift:yshift+h,xshift:xshift+w]=img

    bdrct[:,0]+=xshift
    bdrct[:, 1] += yshift

    cropimg=bkimg[bdrct[0][1]:bdrct[1][1],bdrct[0][0]:bdrct[1][0]]
    return cropimg,xshift,yshift


def img2bin_uint(imgin):
    img=np.array(imgin)
    thres=1
    img[img<thres]=0
    img[img>=thres]=255
    return   img


# ########################################################################################################
def split_imlist(imlist,numsplit):
    numall=len(imlist)
    interval=numall//numsplit
    reslist=[]
    for i in range(0,numsplit):
        if i==numsplit-1:
            reslist.append(imlist[i*interval:])
        else:
            reslist.append(imlist[i*interval:i*interval+interval])
    return  reslist

def jobfunc(imlist,mattingroot,dstroot,id):
    numim=len(imlist)
    for i,im in enumerate(imlist):
        imname = os.path.basename(im)

        imkey = imname.split('.')[0]
        imkeynum = int(imkey)
        mattingim = os.path.join(mattingroot, str(imkeynum).zfill(5) + '.png')

        # print(mattingim)

        img = cv2.imread(im)
        mattingimg = cv2.imread(mattingim)
        mattingimg = cv2.resize(mattingimg, (0, 0), fx=0.25, fy=0.25)
        h, w, c = img.shape
        fusion_img = np.zeros((h, w, 4))
        fusion_img[:, :, 0:3] = img
        fusion_img[:, :, 3] = mattingimg[:, :, 0]
        cv2.imwrite(os.path.join(dstroot, imname.replace('.jpg', '.png')), fusion_img)
        print(str(id)+'  '+str(i) + '/' + str(numim))
def mp_main():
    srcroot=r'/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256'
    mattingroot=r'/home/tao/disk1/Dataset/CelebAHairMask-HQ/V1.0/mask/mask'
    dstroot='/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256-4c'

    ims=get_ims(srcroot)
    numworkers=8
    imlists=split_imlist(ims,numworkers)

    plist=[]
    for i,imlist in enumerate(imlists):
        p = Process(target=jobfunc, args=(imlist,mattingroot,dstroot,i))
        p.start()
        plist.append(p)

    for p in plist:
        p.join()

#########################################################################################################