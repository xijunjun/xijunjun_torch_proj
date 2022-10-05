# coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform, math
from delaunay2D import Delaunay2D
import argparse
import cv2
import torch

from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_alignment
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from facexlib.matting import init_matting_model
from facexlib.utils import img2tensor
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from warp_funcs import *


from scipy import interpolate
from scipy.optimize import curve_fit


key_dic={}
def load_key_val():
    key_val_path='key_val.txt'
    if 'Windows' in platform.system():
        key_val_path='key_val_win.txt'
    lines=open(key_val_path).readlines()
    for line in lines:
        item=line.split(' ')
        vals=item[1].split(',')
        val_lst=[]
        for val in vals:
            val_lst.append(int(val))
        key_dic[item[0]]=val_lst
        # print item[0],val_lst
load_key_val()


def get_ims(imgpath):
    imgpathlst = []
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

face_template_512 = [[200, 245], [315, 245], [256, 370]]

def crop_face_bypt(src_pts, srcimg):
    srctp = np.array(face_template_512, np.float32)
    dsttp = np.array(src_pts, np.float32)
    A = cv2.getAffineTransform(dsttp, srctp)
    res = cv2.warpAffine(srcimg, A, (512, 512))
    return res

def get_rotate_pt(pts):
    theta = 90 / 180.0 * math.pi
    ptsnew = np.array(pts)
    ptsnew -= pts[0]
    pt = ptsnew[1]
    x, y = pt[0], pt[1]
    x2 = (x * math.cos(theta) - y * math.sin(theta))
    y2 = (y * math.cos(theta) + x * math.sin(theta))
    ptsnew[1] = [x2, y2]
    ptsnew += pts[0]

    return ptsnew[1]

def crop_face_by2pt(src_pts, srcimg):
    face_template_512_new = np.array(face_template_512, np.float32) / 4.0
    rt = get_rotate_pt(face_template_512_new[0:2])
    face_template_512_new[2] = np.array(rt)

    src_pts_new = src_pts
    rt = get_rotate_pt(src_pts_new[0:2])
    src_pts_new[2] = np.array(rt)

    dsttp = np.array(src_pts_new, np.float32)
    srctp = np.array(face_template_512_new, np.float32)
    A = cv2.getAffineTransform(dsttp, srctp)
    res = cv2.warpAffine(srcimg, A, (128, 128))
    # cv2.resize(res,(128,128))
    return res

def limit_img_longsize_scale(img, target_size):
    img_ = np.array(img)
    h, w, c = img.shape
    if w > h:
        scale = target_size / w
    else:
        scale = target_size / h
    scale = min(1.0, scale)
    tw, th = int(scale * w), int(scale * h)
    if scale < 1:
        img_ = cv2.resize(img_, (tw, th))
    return img_, scale

def rct2rctwh(rct):
    return np.array([rct[0], rct[1], rct[2] - rct[0], rct[3] - rct[1]])

def rctwh2rct(rct):
    return np.array([rct[0], rct[1], rct[2] + rct[0], rct[3] + rct[1]])

def bigger_rct(wratio, hratio, rctin):
    rct = rct2rctwh(rctin)
    wratio = (wratio - 1.0) * 0.5
    hratio = (hratio - 1.0) * 0.5
    delta_w = (rct[2]) * wratio + 0.5
    delta_h = (rct[3]) * hratio + 0.5
    # return limit_rct([int(rct[0]-delta_w),int(rct[1]-delta_h),int(rct[2]+delta_w*2),int(rct[3]+delta_h*2)],imgshape)
    rct = [rct[0] - delta_w, rct[1] - delta_h, rct[2] + delta_w * 2, rct[3] + delta_h * 2]
    rct = rctwh2rct(rct)
    return rct

def crop_pad(img, bdrct):
    h, w, c = img.shape

    extend_tlx = min(0, bdrct[0][0])
    extend_tly = min(0, bdrct[0][1])
    extend_brx = max(w, bdrct[1][0])
    extend_bry = max(h, bdrct[1][1])
    extendw = extend_brx - extend_tlx
    extendh = extend_bry - extend_tly
    bkimg = np.zeros((extendh, extendw, 3), img.dtype)
    # print('cropimg',extend_tlx,extend_tly)

    xshift = 0 - extend_tlx
    yshift = 0 - extend_tly
    bkimg[yshift:yshift + h, xshift:xshift + w] = img

    bdrct[:, 0] += xshift
    bdrct[:, 1] += yshift

    cropimg = bkimg[bdrct[0][1]:bdrct[1][1], bdrct[0][0]:bdrct[1][0]]
    return cropimg, xshift, yshift

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



def limit_window_auto(winname,imgin):
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
            # img = cv2.resize(img, (tw, th))
            cv2.resizeWindow(winname, tw, th)
    else:
        th = int(sh)
        if th < h:
            tw = int(w * (th / h))
            # img = cv2.resize(img, (tw, th))
            cv2.resizeWindow(winname, tw, th)
    cv2.resizeWindow(winname, w, h)
    # print(tw, th)
    return img


# def get_crop_param(landpts5):
#     template_2048=np.array([[863,1147],[1217,1147],[1043,1383],[889,1547],[1193,1547]])
#
#     warp_param_face_2048=cv2.estimateAffinePartial2D(landpts5, template_2048, method=cv2.LMEDS)[0]
#     return warp_param_face_2048

def get_crop_param(landpts5):
    template_2048=np.array([[863,1147],[1217,1147],[1043,1383],[889,1547],[1193,1547]])
    template_2048 +=244
    warp_param_face_2048=cv2.estimateAffinePartial2D(landpts5, template_2048, method=cv2.LMEDS)[0]
    warp_param_face_inv=cv2.invertAffineTransform(warp_param_face_2048)

    return warp_param_face_2048,warp_param_face_inv


def get_crop_param_targetsize(landpts5,targetsize):
    template_2048=np.array([[863,1147],[1217,1147],[1043,1383],[889,1547],[1193,1547]])
    template_2048 +=244
    template_nrom=template_2048/2536
    head_temp=template_nrom*targetsize

    warp_param_face_2048=cv2.estimateAffinePartial2D(landpts5, head_temp, method=cv2.LMEDS)[0]
    warp_param_face_inv=cv2.invertAffineTransform(warp_param_face_2048)

    return warp_param_face_2048,warp_param_face_inv

# def get_mean(land98,indlist):

def land98to5(land98):
    land98_=np.array(land98)

    pts5=[]
    indlist=list(range(60,69))
    indlist[-1]=96
    x,y=land98_[indlist][:,0].mean(),land98_[indlist][:, 1].mean()
    pts5.append([x,y])

    indlist=list(range(68,76))
    indlist[-1]=97
    x,y=land98_[indlist][:,0].mean(),land98_[indlist][:, 1].mean()
    pts5.append([x,y])

    pts5.append(land98_[54])
    pts5.append(land98_[76])
    pts5.append(land98_[82])

    return  np.array(pts5,np.int32)
    # print(indlist)

def pt_trans(pts,param):

    dst=[]
    for pt in pts:
        x = pt[0]*param[0][0]+pt[1]*param[0][1]+param[0][2]
        y = pt[0] * param[1][0] + pt[1] * param[1][1] + param[1][2]
        dst.append([x,y])
    return  np.array(dst)

def pt_trans_one(pt,param):
    x = pt[0]*param[0][0]+pt[1]*param[0][1]+param[0][2]
    y = pt[0] * param[1][0] + pt[1] * param[1][1] + param[1][2]
    return  np.array([x,y])

def image_1to3c(imagein):
    image3c=np.array(imagein[:,:,None])
    image3c=image3c.repeat(3,axis=2)
    return  image3c

def pred_seg_bise(bise_net,img_input):

    # img_input = cv2.imread(img_path)
    h,w,c=img_input.shape
    img_input = cv2.resize(img_input, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img2tensor(img_input.astype('float32') / 255., bgr2rgb=True, float32=True)
    normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
    img = torch.unsqueeze(img, 0).cuda()

    with torch.no_grad():
        out = bise_net(img)[0]
    parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],#19
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    # 0: 'background'
    # attributions = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
    #                 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose',
    #                 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l',
    #                 16 'cloth', 17 'hair', 18 'hat']
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    stride=1
    vis_parsing_anno = cv2.resize(vis_parsing_anno, (w,h),interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # vis_im = cv2.addWeighted(img, 0.4, vis_parsing_anno_color, 0.6, 0)]

    return  vis_parsing_anno_color

def get_target_seg(segimgin,color):

    segbin=segimgin==color
    segbin=np.all(segbin,axis=2,keepdims=True)
    segbin=(segbin*255).astype(np.uint8)

    # segbin=image_1to3c(segbin)
    return  segbin

def get_mat(matnet,imgin):
    # read image
    img = np.array(imgin)/ 255.
    # unify image channels to 3
    if len(img.shape) == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, 0:3]

    img_t = img2tensor(img, bgr2rgb=True, float32=True)
    normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img_t = img_t.unsqueeze(0).cuda()

    # resize image for input
    _, _, im_h, im_w = img_t.shape
    ref_size = 512
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    img_t = F.interpolate(img_t, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = matnet(img_t, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return (matte * 255).astype('uint8')

def get_calva_bottom(land98):
    eyeindlist=[]
    eyeindlist.extend(list(range(60,68)))
    eyeindlist.append(96)
    eyeindlist.extend(list(range(68,76)))
    eyeindlist.append(97)
    lands_eye=np.array(land98)[eyeindlist]
    # maxy=np.min(lands_eye[:,1])
    maxy = np.max(lands_eye[:, 1])

    maxy=land98[57][1]
    return  maxy

def img2bin_uint(imgin):
    img=np.array(imgin)
    thres=100
    img[img<thres]=0
    img[img>=thres]=255
    return   img

def simplify_mask(maskin):

    contours, h = cv2.findContours(maskin.copy()[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    # img = cv2.drawContours(maskin.copy(), [contours[0]], -1, (0, 255, 0), 5)
    img = cv2.drawContours(np.zeros_like(maskin), [contours[0]], -1, (255, 255, 255), -1)

    return img

def crop_pad(img,bdrct):
    h,w,c=img.shape

    extend_tlx = min(0,bdrct[0][0])
    extend_tly = min(0, bdrct[0][1])
    extend_brx = max(w,bdrct[1][0])
    extend_bry = max(h, bdrct[1][1])
    extendw=extend_brx-extend_tlx
    extendh = extend_bry - extend_tly
    bkimg=np.zeros((extendh,extendw,3),img.dtype)+127
    # print('cropimg',extend_tlx,extend_tly)

    xshift=0-extend_tlx
    yshift = 0 - extend_tly
    bkimg[yshift:yshift+h,xshift:xshift+w]=img

    bdrct[:,0]+=xshift
    bdrct[:, 1] += yshift

    cropimg=bkimg[bdrct[0][1]:bdrct[1][1],bdrct[0][0]:bdrct[1][0]]
    return cropimg,xshift,yshift

def get_crop_rct(matrct):
    x,y,w,h=tuple(matrct)
    x2=x+w
    y2=y+h
    # ctx = x + w / 2
    # cty=y+h/2
    # left and right
    deltaw=0.1*w*0.5
    leftx=x-deltaw
    rightx=x2+deltaw
    neww=rightx-leftx
    #up
    # deltah=0.3*h
    # upy=cty-h/2-deltah
    newh=neww/1536*1280
    upy=y2-newh

    croprct=[leftx,upy,rightx,y2]
    croprct=np.array(croprct,np.int32)
    return  croprct

def get_face_land_rct(det_net,align_net,framein):
    with torch.no_grad():
        frame=np.array(framein)
        img_scale, scale = limit_img_longsize_scale(frame, 512)
        bboxes = det_net.detect_faces(img_scale, 0.97) / scale

        bboxes=list(bboxes)
        bboxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),reverse=True)

        landmark_list=[]
        bbox_list=[]
        for box in bboxes:
            rct = box[0:4].astype(np.int32)
            land5 = box[5:5 + 10].reshape((5, 2)).astype(np.int32)
            all_face_rcts.append(rct)
            exd_rct = np.array(bigger_rct(1.1, 1.1, rct), np.int32)
            faceimg, xshift, yshift = crop_pad(frame, exd_rct.reshape(2, 2))
            cv2.rectangle(frame, (exd_rct[0], exd_rct[1]), (exd_rct[2], exd_rct[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (rct[0], rct[1]), (rct[2], rct[3]), (0, 0, 255), 2)
            landmarks = align_net.get_landmarks(faceimg) + np.array([exd_rct[0], exd_rct[1]])
            landmarks = np.array(landmarks, np.int32)
            landmark_list.append(landmarks)
            bbox_list.append(rct)

    return bbox_list,landmark_list

def get_calva_crop_rct(imagein,matimgin,landmarksin):
    image=imagein.copy()
    matimg=matimgin.copy()
    landmarks=np.array(landmarksin)

    calvaw=1536
    calvah=1280
    upgap=180
    lrgap=60
    limitw=calvaw-lrgap*2
    limith=calvah-upgap
    limit_whratio=limitw/limith

    matbin = img2bin_uint(matimg)
    matbin_sim = simplify_mask(matbin)

    calva_bottom_y=int(landmarks[54][1])

    h, w, c = matbin_sim.shape
    matbin_sim[calva_bottom_y:h, :, :] = 0
    matbin_sim = simplify_mask(matbin_sim)
    matrct = cv2.boundingRect(matbin_sim[:, :, 0])
    # croprct = get_crop_rct(matrct)
    x0,y0,w,h=tuple(matrct)
    x1=x0+w
    y1=y0+h
    whratio=w/h

    # calva_limit_rct=matrct
    if whratio>limit_whratio:
        targeth=w/limit_whratio
        deltah=targeth-h
        calva_limit_rct=[x0,y0-deltah,x1,y1]
    else:
        targetw=h*limit_whratio
        deltaw=targetw-w
        calva_limit_rct = [x0-deltaw, y0 , x1+deltaw, y1]

    calva_limit_rct=np.array(calva_limit_rct)

    calva_crop_rct=[calva_limit_rct[0]-lrgap,calva_limit_rct[1]-upgap,calva_limit_rct[2]+lrgap,calva_limit_rct[3]]

    calva_limit_rct=np.array(calva_limit_rct,np.int32)
    calva_crop_rct=np.array(calva_crop_rct,np.int32)

    return  calva_limit_rct,calva_crop_rct


def get_halfhead_crop_rct(landmarks,headw):

    h, w, c = headw,headw,3
    calvaw=headw
    calvah=int(headw*10/16)

    print(calvaw,calvah)

    limit_whratio=1.0*calvaw/calvah
    calva_bottom_y=int(landmarks[54][1])
    upy=calva_bottom_y-calvah
    # half_head_rct=[[0,upy],[headw,calva_bottom_y]]

    half_head_rct = [0, upy, headw, calva_bottom_y]
    half_head_rct=np.array(half_head_rct,np.int32)

    half_headh, half_headw=calvah,calvaw

    return half_head_rct,half_headh,half_headw



class Periodic_list(object):

    def __init__(self,listin):
        self.mylist=listin
        self.numpts=self.mylist.shape[0]

    def __getitem__(self, ind):
        if ind<0:
            ind+=self.numpts
        ind = ind % self.numpts
        return  self.mylist[ind]

def  ind_trans(mylist,indlist):
    numpts = mylist.shape[0]
    numind=len(indlist)
    trans_indlist=[]
    for i,ind in enumerate(indlist):
        if ind < 0:
            ind += numpts
        ind = ind % numpts
        trans_indlist.append(ind)
    return  trans_indlist

    # # Colors for all 20 parts
    # part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
    #                [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
    #                [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
    #                [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    # # 0: 'background'
    # # attributions = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
    # #                 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose',
    # #                 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l',
    # #                 16 'cloth', 17 'hair', 18 'hat']

part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
               [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
               [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
               [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]

def smo_the_pts(cont_ptlist,meannum):
    cont_input=np.array(cont_ptlist)
    cont_smo=np.array(cont_ptlist)
    numpts = cont_input.shape[0]

    for i in range(0, numpts):
        indlist=range(i- meannum//2,i+meannum//2)
        indlist=ind_trans(cont_input, indlist)
        cont_smo[i][0] = int(cont_input[indlist,0].mean())
        cont_smo[i][1] = int(cont_input[indlist, 1].mean())
    contlist_pd=cont_smo
    return  contlist_pd

def smo_the_pts_close(cont_ptlist,meannum):
    cont_input=np.array(cont_ptlist)
    cont_smo=np.array(cont_ptlist)
    numpts=cont_input.shape[0]
    # meannum=400
    for i in range(0, numpts):
        indlist=range(i- meannum//2,i+meannum//2)
        indlist=ind_trans(cont_input, indlist)
        cont_smo[i][0] = int(cont_input[indlist,0].mean())
        cont_smo[i][1] = int(cont_input[indlist, 1].mean())
    contlist_pd=cont_smo

    return  contlist_pd


def small_to_big(img):
    h,w,c=img.shape
    bkimg=np.zeros((h*2,w*2,3),img.dtype)
    bkimg[h//2:h//2+h,w//2:w//2+w]=img.copy()
    return  bkimg
def big_to_small(img):
    h, w, c = img.shape
    h=h//2
    w=w//2
    return  img[h//2:h//2+h,w//2:w//2+w].copy()

def euclidean(pt1,pt2):
    return  np.linalg.norm(np.array(pt1) - np.array(pt2))

def draw_pts(img,ptlist,r,color,thick,wait=0):
    for pt in ptlist:
        # print(pt)
        cv2.circle(img,tuple(np.array(pt,np.int32)),r,color,thick)
        if wait!=0:
            cv2.imshow('calva_mat_new', limit_img_auto(img))
            cv2.waitKey(wait)

def min_dis_ind(pt1,ptlist):
    dis_list=[]
    for pt in ptlist:
        dis_list.append(euclidean(pt,pt1))
    dis_list=np.array(dis_list)
    ind=np.argmin(dis_list)
    return  ind

def min_disx_ind(pt1,ptlist):
    dis_list=[]
    for pt in ptlist:
        dis_list.append(abs(pt[0]-pt1[0]))
    dis_list=np.array(dis_list)
    ind=np.argmin(dis_list)
    return  ind

def split_cont_by_two_pts(ptlist,pt1,pt2):
    ptlist_np=np.array(ptlist)
    lind=np.where(np.all(ptlist_np==pt1,axis=1))[0][0]
    rind = np.where(np.all(ptlist_np==pt2, axis=1))[0][0]

    numpts=len(ptlist)
    minind=min(lind,rind)
    maxind=max(lind,rind)

    sublist1=ptlist_np[minind:maxind]
    sublist2=[]
    sublist2.extend(ptlist_np[maxind:numpts])
    sublist2.extend(ptlist_np[0:minind])

    if len(sublist2)<1 or len(sublist1)<1:
        return None,None

    # 确定上下
    sub1meany=np.array(sublist1)[:,1].mean()
    sub2meany = np.array(sublist2)[:, 1].mean()
    if sub1meany<sub2meany:
        sublist_up=list(sublist1)
        sublist_down = list(sublist2)
    else:
        sublist_up=list(sublist2)
        sublist_down = list(sublist1)
    # 调整点位的左右顺序
    if sublist_up[0][0]>sublist_up[-1][0]:
        sublist_up.reverse()
    if sublist_down[0][0]>sublist_down[-1][0]:
        sublist_down.reverse()

    # print('+++',sublist_up[0][0],sublist_up[-1][0])
    # print('---', sublist_down[0][0], sublist_down[-1][0])

    return  sublist_up,sublist_down

def get_cont_up_and_down(calva_mat,meank=300):

    h,w,c=calva_mat.shape
    # calva_mat_new=np.array(calva_mat)
    # calva_mat_bin=img2bin_uint(calva_mat)
    calva_mat_bin = np.array(calva_mat)

    bottom_mat=np.array(calva_mat_bin)
    # bottom_mat[h-10:h,:,:]=0
    bottom_mat[0:h - 10, :, :] = 0
    if bottom_mat[:,:,0].sum()<100:
        return None,None

    contoursbt, _ = cv2.findContours(bottom_mat[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursbt.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    ptlistbt=[]
    for pt in contoursbt[0]:
        ptlistbt.append(pt[0])


    contours, _ = cv2.findContours(calva_mat_bin[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    ptlist=[]
    for pt in contours[0]:
        ptlist.append(pt[0])
    ptlist=smo_the_pts(ptlist,meank)
    ptlist = smo_the_pts(ptlist, meank)

    # ##########################
    ch, cw, cc = calva_mat.shape
    rct = cv2.boundingRect(calva_mat_bin[:, :, 0])
    pt_bl = [rct[0], ch]
    pt_br = [rct[0] + rct[2], ch]
    cornet_lb=[rct[0],h]
    corner_rb=[rct[0] + rct[2],h]

    cornet_lb=ptlistbt[min_dis_ind(cornet_lb, ptlistbt)]
    corner_rb = ptlistbt[min_dis_ind(corner_rb, ptlistbt)]

    # cornet_lb=[0,h]
    # corner_rb=[w,h]

    draw_pts(bottom_mat,list([cornet_lb,corner_rb]),10,(255,0,0),10)
    lind=min_dis_ind(cornet_lb, ptlist)
    rind = min_dis_ind(corner_rb, ptlist)

    draw_pts(bottom_mat, list([cornet_lb, corner_rb]), 10, (255, 0, 0), 10)
    draw_pts(bottom_mat, list([ptlist[lind], ptlist[rind]]), 20, (0, 255, 0), 3)
    # cv2.imshow('bottom_mat', limit_img_auto(bottom_mat))

    sublist1, sublist2 = split_cont_by_two_pts(ptlist, ptlist[lind], ptlist[rind])

    if sublist2 is None or sublist1 is None:
        return None,None

    return list(sublist1),list(sublist2)

def get_etou_cont_up_and_down(calva_mat,land,meank=300):

    etou_bty=int(min(land[0][1],land[32][1]))

    h,w,c=calva_mat.shape
    # calva_mat_new=np.array(calva_mat)
    # calva_mat_bin=img2bin_uint(calva_mat)
    calva_mat_bin = np.array(calva_mat)
    calva_mat_bin[etou_bty:h,:,:]=0

    bottom_mat=np.array(calva_mat_bin)
    # bottom_mat[h-10:h,:,:]=0
    bottom_mat[0:etou_bty - 10, :, :] = 0
    if bottom_mat[:,:,0].sum()<100:
        return None,None

    contoursbt, _ = cv2.findContours(bottom_mat[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursbt.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    ptlistbt=[]
    for pt in contoursbt[0]:
        ptlistbt.append(pt[0])

    contours, _ = cv2.findContours(calva_mat_bin[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    ptlist=[]
    for pt in contours[0]:
        ptlist.append(pt[0])
    ptlist=smo_the_pts(ptlist,meank)
    ptlist = smo_the_pts(ptlist, meank)

    # ##########################
    ch, cw, cc = calva_mat.shape
    rct = cv2.boundingRect(calva_mat_bin[:, :, 0])
    pt_bl = [rct[0], ch]
    pt_br = [rct[0] + rct[2], ch]
    cornet_lb=[rct[0],h]
    corner_rb=[rct[0] + rct[2],h]

    cornet_lb=ptlistbt[min_dis_ind(cornet_lb, ptlistbt)]
    corner_rb = ptlistbt[min_dis_ind(corner_rb, ptlistbt)]
    # cornet_lb=[0,h]
    # corner_rb=[w,h]
    draw_pts(bottom_mat,list([cornet_lb,corner_rb]),10,(255,0,0),10)
    lind=min_dis_ind(cornet_lb, ptlist)
    rind = min_dis_ind(corner_rb, ptlist)


    draw_pts(bottom_mat, list([cornet_lb, corner_rb]), 10, (255, 0, 0), 10)
    draw_pts(bottom_mat, list([ptlist[lind], ptlist[rind]]), 20, (0, 255, 0), 3)
    # cv2.imshow('bottom_mat', limit_img_auto(bottom_mat))

    sublist1, sublist2 = split_cont_by_two_pts(ptlist, ptlist[lind], ptlist[rind])
    if sublist1 is None or sublist1 is None:
        return  None,None

    return list(sublist1),list(sublist2)

def half2full(img):
    h,w,c=img.shape
    imgfull=np.zeros((h+h//2,w,3),dtype=img.dtype)+127
    imgfull[0:h,0:w]=img.copy()
    return  imgfull


def get_calva_landmark(align_net,img):
    landmark=0
    fullimg=half2full(img)
    landmark = align_net.get_landmarks(fullimg)
    # cv2.imshow('full',limit_img_auto(fullimg))

    # print(landmark )
    return  landmark

def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0

def to_int32(ptlist):
    return  np.array(ptlist,np.int32)

def get_pt(ptlist,ray):
    angle_list=[]
    for pt in ptlist:
        angle=dot_product_angle(pt-ray[0],ray[1]-ray[0])
        angle_list.append(angle)
    minind=np.argmin(angle_list)
    pt=ptlist[minind]
    return  pt

def get_expand_pts(exp_base_pts,contpts,bottom_pts):
    numdiv=8
    pt_base = exp_base_pts[0]
    leftray = [pt_base,bottom_pts[0]]
    rightray = [pt_base, bottom_pts[1]]
    angle_bt=dot_product_angle(bottom_pts[0]-pt_base, bottom_pts[1]-pt_base)
    angle_left=360-angle_bt
    delta_angle=angle_left/numdiv

    exp_pts_res=[]
    for i in range(1,numdiv):
        rot_mat = cv2.getRotationMatrix2D(tuple(pt_base), -delta_angle*i, 10.5)
        curpt=np.array(pt_trans_one(bottom_pts[0],rot_mat),np.int32)
        exp_pt=get_pt(list(contpts), [pt_base,curpt])
        exp_pts_res.append(exp_pt)

    return  exp_pts_res

def get_expand_pts_divy(contpts,halfhead_mat_3c):
    h,w,c=halfhead_mat_3c.shape

    numdiv=3
    contpts_np=np.array(contpts,np.int32)
    numcnt=len(contpts)
    ind_maxy=np.argmin(contpts_np[:,1])
    left_cont_np=contpts_np[0:ind_maxy]
    right_cont_np = contpts_np[ind_maxy:numcnt]
    halfhead_topy=contpts_np[ind_maxy][1]
    halfhead_height=h-halfhead_topy
    dgap=halfhead_height/(numdiv+1)

    expand_ptlist=[]
    ylist=[]
    for i in range(0,numdiv):
        ylist.append(halfhead_topy+dgap*(i+1))

    for i in range(0, numdiv):
        leftysub=left_cont_np[:,1]-ylist[i]
        leftysub=np.abs(leftysub)
        tplind = np.argmin(leftysub)
        expand_ptlist.append(left_cont_np[tplind])
    expand_ptlist.reverse()
    expand_ptlist.append(contpts_np[ind_maxy])

    for i in range(0, numdiv):
        rightysub=right_cont_np[:,1]-ylist[i]
        rightysub=np.abs(rightysub)
        tprind = np.argmin(rightysub)
        expand_ptlist.append(right_cont_np[tprind])

    return  expand_ptlist


def get_expand_pts_divd(contpts,halfhead_mat_3c):
    h,w,c=halfhead_mat_3c.shape

    numdiv=15
    contpts_np=np.array(contpts,np.int32)
    numcnt=len(contpts)

    ind_gap=int(numcnt/numdiv)
    expand_ptlist=contpts_np[::ind_gap]
    return  list(expand_ptlist)

def get_portrait_extland(contpts,numpts=20):

    # numpts=20
    numdiv=numpts-1
    contpts_np=np.array(contpts,np.int32)
    numcnt=len(contpts)

    ind_gap=int(numcnt/numdiv)
    # expand_ptlist=contpts_np[::ind_gap]
    expand_ptlist=[]
    expand_ptlist.append(contpts_np[0])

    for i in range(0,numpts-2):
        expand_ptlist.append(contpts_np[(i+1)*ind_gap])
    expand_ptlist.append(contpts_np[-1])

    return  list(expand_ptlist)


def expand_the_pts(exp_base_pts, exp_pts):
    base_pt = exp_base_pts[0]
    exp_pt_result = []
    exp_ratio=1.15
    shape_param=[0.3,0.7,0.9,1.0]
    tp_param=list(shape_param[0:3])
    tp_param.reverse()
    shape_param.extend(tp_param)
    # print(shape_param)
    for i, ept in enumerate(exp_pts):
        lenth = euclidean(ept, base_pt)
        xpt = [lenth, 0]
        trans_param = cv2.estimateAffinePartial2D(np.array([base_pt, ept]), np.array([[0, 0], xpt]), method=cv2.LMEDS)[0]
        trans_param_inv = cv2.invertAffineTransform(trans_param)
        exp_dis=exp_ratio*lenth-lenth
        xpt = pt_trans_one([lenth+exp_dis * shape_param[i], 0], trans_param_inv)
        exp_pt_result.append(np.array(xpt, np.int32))
    return exp_pt_result




keylist=[0]
def show_img_two(img1,img2):
    flag=0
    while 1:
        if flag==0:
            cv2.imshow('imx',limit_img_auto(np.concatenate([img1,img2],axis=1)))
            flag=1
        else:
            cv2.imshow('imx', limit_img_auto(np.concatenate([img2, img1], axis=1)))
            flag=0
        keylist[0]=cv2.waitKey(0)
        if keylist[0]==13:
            break
        if keylist[0]==27:
            exit(0)

# def get_hair():




def get_landmask_mask(image,land):
    land=np.array(land,np.int32)
    # indlist=list(land[0:33])
    # indlist.extend(list(land[[46,45,44,43,42,37,36,35,34,33]]))

    indlist = list([0,10,22,32,44,35])

    landlist=land[indlist]
    landmask=np.zeros_like(image)

    cv2.fillConvexPoly(landmask, np.array(landlist), (255, 255, 255))

    return landmask

# def callback_getetoupts():
#     cv2.namedWindow('preview', cv2.WINDOW_FREERATIO)
#     cv2.moveWindow('preview',0,0)
#     bdrct= cv2.boundingRect(np.array(ptlist))
#     delta_w=int(bdrct[2]*0.3)
#     delta_h=int(bdrct[3]*0.5)
#     pt_tl=[bdrct[0],bdrct[1]]
#     pt_rd=[bdrct[0]+ bdrct[2]-1,bdrct[1]+ bdrct[3]-1]
#     pt_tl[0]=0 if pt_tl[0]-delta_w<0 else pt_tl[0]-delta_w
#     pt_rd[0] =ori_img.shape[1]-1 if  pt_rd[0]+delta_w>ori_img.shape[1]-1 else pt_rd[0]+delta_w
#     pt_tl[1] =0 if pt_tl[1]-delta_h<0 else pt_tl[1]-delta_h
#     pt_rd[1] = ori_img.shape[0]-1 if pt_rd[1]+delta_h>ori_img.shape[0]-1 else pt_rd[1]+delta_h
#     pltimg = ori_img[pt_tl[1]:pt_rd[1]+1, pt_tl[0]:pt_rd[0]+1].copy()
#     refresh_preview(pltimg, pt_tl)
#     while 1:
#         key=cv2.waitKey(0)
#         if key in key_dic['ENTER']:
#             break
#         if key in key_dic['SPACE']:
#             global_var[1]=(global_var[1]+1)%4
#         if key in key_dic['UP']:
#             if ptlist[global_var[1]][1]-pt_tl[1]-global_var[2]>=0:
#                 ptlist[global_var[1]][1] -=global_var[2]
#         if key in key_dic['DOWN']:
#             if ptlist[global_var[1]][1]-pt_tl[1]+global_var[2]<=pltimg.shape[0]-1:
#                 ptlist[global_var[1]][1]  +=global_var[2]
#         if key in key_dic['LEFT']:
#             if ptlist[global_var[1]][0]-pt_tl[0]-global_var[2]>=0:
#                 ptlist[global_var[1]][0]  -=global_var[2]
#         if key in key_dic['RIGHT']:
#             if ptlist[global_var[1]][0]-pt_tl[0]+global_var[2]<=pltimg.shape[1]-1:
#                 ptlist[global_var[1]][0]  +=global_var[2]
#         if key in key_dic['ESC']:
#             del ptlist[:]
#             cv2.destroyWindow('preview')
#             return
#         refresh_preview(pltimg, pt_tl)
#         # refresh_ori()
#     # print ptlist
#     platelist.append(list(ptlist))#list.append([])只拷贝索引,不拷贝对象
#     refresh_ori()
#     cv2.destroyWindow('preview')


def line_ptlist(image,ptlistin,color,thick):
    ptlist=list(np.array(ptlistin,np.int32))

    numpt=len(ptlist)
    for i,pt in enumerate(ptlist):
        if i==numpt-1:
            break
        cv2.line(image,tuple(ptlist[i]),tuple(ptlist[i+1]),color,thickness=thick,lineType=cv2.LINE_AA)
        # cv2.line(image, tuple(ptlist[i]), tuple(ptlist[i + 1]), color, thickness=thick, lineType=cv2.LINE_8)




def func(x, a, b, c):
  return b * np.power(x, a) + c


global global_etou_pts
global global_val
global global_numetou
global_etou_pts=[]
global_val=[0,0,0,0]
global_numetou=20



def visual_ano():
    halfheadalign_visual=np.array(halfheadalign)
    draw_pts(halfheadalign_visual, list(global_etou_pts), 10, (255, 0, 255), 5)

    num_global_etou_pts=len(global_etou_pts)

    if num_global_etou_pts>0:
        pt_cursor_ind=global_val[0]
        draw_pts(halfheadalign_visual, list([global_etou_pts[pt_cursor_ind]]), 10, (255, 0, 255), 5)
        draw_pts(halfheadalign_visual, list([global_etou_pts[pt_cursor_ind]]), 10, (0, 0, 255), 10)

    global_etou_pts_np=np.array(global_etou_pts)
    if num_global_etou_pts > 5:

        # x = global_etou_pts_np[:,0]
        # y = global_etou_pts_np[:, 1]
        # # cs = interpolate.CubicSpline(x, y, bc_type='periodic')
        # cs = interpolate.CubicSpline(x, y)
        # xx = np.linspace(global_etou_pts[0][0], global_etou_pts[-1][0], 100)
        # yy=cs(xx)
        # fitpts=[xx,yy]
        # fitpts_np=np.array(fitpts).T
        # np.swapaxes(fitpts_np,0,1).reshape(-1,2)
        # draw_pts(halfheadalign_visual, list(fitpts_np), 4, (0, 0, 255), 4)
        # line_ptlist(halfheadalign_visual, list(fitpts_np), (0, 0, 255),4)

        line_ptlist(halfheadalign_visual, list(global_etou_pts_np), (0, 0, 255), 4)




    cv2.imshow('halfheadalign', halfheadalign_visual)


def interpolate_pts(isup):
    global global_etou_pts
    global global_val
    global global_numetou
    bakimg=np.zeros_like(halfheadalign)
    line_ptlist(bakimg, list(global_etou_pts), (255, 255, 255), 3)

    contoursbt, _ = cv2.findContours(bakimg[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursbt.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    ptlist=[]
    for pt in contoursbt[0]:
        ptlist.append(pt[0])

    lind=min_dis_ind(global_etou_pts[0], ptlist)
    rind = min_dis_ind(global_etou_pts[-1], ptlist)
    sublistup, sublistdown = split_cont_by_two_pts(ptlist, ptlist[lind], ptlist[rind])

    if isup:
        intpts=get_portrait_extland(sublistup, global_numetou)
    else:
        intpts=get_portrait_extland(sublistdown, global_numetou)

    intpts[0]=global_etou_pts[0]
    intpts[-1] = global_etou_pts[-1]
    global_etou_pts=list(intpts)


def add_pts(event,x,y,flags,param):
    global global_etou_pts
    global global_val
    global global_numetou
    # if len(ptlist) == 4:
    #     return
    if event == cv2.EVENT_LBUTTONDOWN:
        # global_etou_pts[0]=[x,y]
        if len(global_etou_pts)<global_numetou:
            global_etou_pts.append([x,y])
        else:
            global_val[0] = min_dis_ind([x, y], global_etou_pts)
            global_etou_pts[global_val[0]]=[x,y]

        visual_ano()
    if event==cv2.EVENT_MOUSEMOVE:
        if len(global_etou_pts) == global_numetou:
            global_val[0] =min_dis_ind([x,y],global_etou_pts)
            visual_ano()




        # ptlist.append([x,y])
        # refresh_ori()
        # if len(ptlist)==4:
        #     get4pts()
        #     get_info()
        #     refresh_ori()
        #     cv2.waitKey(1)#注意此处等待按键


# def add_pts():


def get_etoupts():
    cv2.setMouseCallback('halfheadalign', add_pts)

    while 1:
        key=cv2.waitKey(0)
        pt_cursor_ind = global_val[0]
        deltamove=2

        print('key:',key)
        # if key in key_dic['ENTER']:
        #     break
        if key in key_dic['SPACE']:
            global_val[0]=0 if len(global_etou_pts)==0 else (pt_cursor_ind+1)%len(global_etou_pts)
        if key in [ord('b')]:
            global_val[0]=0 if len(global_etou_pts)==0 else (len(global_etou_pts)+pt_cursor_ind-1)%len(global_etou_pts)

        if key in key_dic['UP']:
            global_etou_pts[pt_cursor_ind][1]-=deltamove
        if key in key_dic['DOWN']:
            global_etou_pts[pt_cursor_ind][1] += deltamove
        if key in key_dic['LEFT']:
            global_etou_pts[pt_cursor_ind][0] -= deltamove
        if key in key_dic['RIGHT']:
            global_etou_pts[pt_cursor_ind][0] += deltamove
        if key in [ord('o')]:
            interpolate_pts(True)
        if key in [ord('p')]:
            interpolate_pts(False)
            # print(key)

        # if key in key_dic['ESC']:
        #     del ptlist[:]
        #     cv2.destroyWindow('preview')
        #     return
        if key in key_dic['BACK']:
            if   len(global_etou_pts)>=1:
                global_etou_pts.pop(-1)
                global_val[0]= 0 if len(global_etou_pts)==0 else (pt_cursor_ind)%len(global_etou_pts)

        if key in key_dic['ESC']:
            exit(0)

        if key in key_dic['ENTER']:
            if global_numetou==len(global_etou_pts):
                break

        visual_ano()






# pt_cursor_ind
def get_imkey_ext(impath):
    imname=os.path.basename(impath)
    imkey=imname.split('.')[0]
    ext=imname.replace(imkey,'')
    return imkey,ext



if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)
    matnet = init_matting_model()
    bise_net = init_parsing_model(model_name='bisenet')

    # srcroot = '/home/tao/mynas/Dataset/FaceEdit/sumiao/'
    # srcroot=r'/home/tao/mynas/Dataset/hairforsr/femalehd'
    # srcroot=r'/home/tao/Downloads/image_unsplash'
    # srcroot=r'/home/tao/Pictures/imupsplash'
    # srcroot = r'/home/tao/Pictures/imtest'
    # srcroot=r'/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/ffhq'

    # srcroot = r'/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/ffhq2k'
    srcroot=r'/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/pexel_side_face'

    dstroot = '/home/tao/disk1/Dataset/Project/FaceEdit/taobao_sumiao/crop/'
    dstroot=r'/home/tao/mynas/Dataset/FaceEdit/image_unsplash_dst'

    # srcroot='/home/tao/mynas/Dataset/FaceEdit/sumiao'



    ims = get_ims(srcroot)
    # face_size = 2048
    # head_size = 2536
    head_size = 2048


    for i, im in enumerate(ims):
        imkey, ext=get_imkey_ext(im)
        print(imkey, ext)
        txtpath=os.path.join(srcroot,imkey+'_calvaland.txt')
        if os.path.exists(txtpath):
            continue


        all_face_rcts = []
        all_face_lands = []
        frame = cv2.imread(im)
        image_const=np.array(frame)
        bbox_list, landmark_list = get_face_land_rct(det_net, align_net, image_const)

        landmark_list=[landmark_list[0]]
        for j,landmarks in enumerate(landmark_list):




            frame_vis = np.array(frame)
            ##########裁剪出单张人头
            land5_from98=land98to5(landmarks)
            # warp_param_head,warp_param_face_inv=get_crop_param(land5_from98)
            warp_param_head,warp_param_face_inv=get_crop_param_targetsize(land5_from98, head_size)

            headalign = cv2.warpAffine(image_const, warp_param_head, (head_size, head_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
            headalign_vis=np.array(headalign)
            h,w,c=headalign.shape

            print('headalign.shape：',headalign.shape)

            # print('facealign.shape ',facealign.shape)

            ##########获取单张人头的matting和seg结果
            head_mat = get_mat(matnet, headalign)
            head_seg_bise = pred_seg_bise(bise_net, headalign)
            head_mat_3c = image_1to3c(head_mat)

            ##########获取单张人头的关键点
            land98_in_crop=pt_trans(landmarks,warp_param_head)
            # for pt in land98_in_crop:
            #     pt=np.array(pt,np.int32)
            #     cv2.circle(headalign, (pt[0], pt[1]), 10, (255, 0, 0), -1, -1)

            headlandmask = get_landmask_mask(headalign, land98_in_crop)

            ##########获取颅顶裁剪框
            calva_bottom_y=int(get_calva_bottom(land98_in_crop))

            halfhead_crop_rct,half_headh,half_headw=get_halfhead_crop_rct(land98_in_crop, head_size)
            # calva_limit_rct, calva_crop_rct = get_calva_crop_rct(headalign, head_mat_3c, land98_in_crop)

            halfhead_quad_incrop=[[halfhead_crop_rct[0],halfhead_crop_rct[1]],[halfhead_crop_rct[2],halfhead_crop_rct[1]],
                           [halfhead_crop_rct[2],halfhead_crop_rct[3]],[halfhead_crop_rct[0],halfhead_crop_rct[3]]]
            halfhead_quad_incrop=np.array(halfhead_quad_incrop,np.float32)

            halfhead_quad_inv=pt_trans(list(halfhead_quad_incrop) ,warp_param_face_inv)
            halfhead_quad_inv=np.array(halfhead_quad_inv,np.int32)

            halfhead_dst_quad = np.array([[0, 0], [half_headw, 0], [half_headw, half_headh], [0, half_headh]])
            param_halfhead_incrop = cv2.estimateAffinePartial2D(halfhead_quad_incrop, halfhead_dst_quad, method=cv2.LMEDS)[0]

            halfheadalign = cv2.warpAffine(headalign, param_halfhead_incrop, (half_headw, half_headh), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
            halfhead_mat_3c = cv2.warpAffine(head_mat_3c, param_halfhead_incrop, (half_headw, half_headh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            halfhead_seg_bise= cv2.warpAffine(head_seg_bise, param_halfhead_incrop, (half_headw, half_headh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            param_halfhead_inori = cv2.estimateAffinePartial2D(halfhead_quad_inv, halfhead_dst_quad, method=cv2.LMEDS)[0]
            param_halfhead_inori_inv=cv2.invertAffineTransform(param_halfhead_inori)
            landmark_in_half=pt_trans(landmarks,param_halfhead_inori)

            calva_mat_bin = img2bin_uint(halfhead_mat_3c)

            ##############构造控制点
            # get_ctl_pts(small_to_big(calva_croped), small_to_big(calva_seg), small_to_big(calva_mat))
            # get_ctl_pts(calva_croped, calva_seg, calva_mat)


            calva_mat_bin=cv2.erode(calva_mat_bin ,kernel=np.ones((19,19),np.uint8),iterations=2)
            calva_mat_bin = cv2.dilate(calva_mat_bin, kernel=np.ones((19, 19), np.uint8), iterations=2)

            calva_mat_bin = cv2.dilate(calva_mat_bin, kernel=np.ones((9, 9), np.uint8), iterations=2)
            calva_mat_bin=cv2.erode(calva_mat_bin ,kernel=np.ones((9,9),np.uint8),iterations=2)


            cv2.imshow('calva_mat_bin',limit_img_auto(calva_mat_bin))

            up_cont_pts,down_cont_pts=get_cont_up_and_down(calva_mat_bin,3)
            ##底部两个关键点


            ch,cw,cc=halfhead_mat_3c.shape
            rct = cv2.boundingRect(calva_mat_bin[:, :, 0])
            pt_bl = [rct[0], ch]
            pt_br = [rct[0] + rct[2], ch]
            Calva_bottom_pts = [pt_bl, pt_br]


            ##扩张基准点
            Calva_base_pts=[landmark_in_half[51]]
            ####构造扩张点
            Calva_expand_pts=get_expand_pts(Calva_base_pts, up_cont_pts, Calva_bottom_pts)
            Calva_expand_pts_result = expand_the_pts(Calva_base_pts, Calva_expand_pts)


            # Calva_top_pts=get_expand_pts_divy(up_cont_pts,halfhead_mat_3c)
            # Calva_top_pts=get_expand_pts_divd(up_cont_pts, halfhead_mat_3c)
            Calva_top_pts=get_portrait_extland(up_cont_pts)

            print('Calva_top_pts:',len(Calva_top_pts))

            draw_pts(halfheadalign, list(merge_pts([Calva_bottom_pts,Calva_base_pts,Calva_expand_pts])), 20, (0, 0,255), 2)
            draw_pts(halfheadalign, list(Calva_top_pts), 10, (0, 255, 0), 5)

            halfhairmask=get_target_seg(halfhead_seg_bise,part_colors[17])
            halfheadmask=255-get_target_seg(halfhead_seg_bise,[255,255,255])
            facemask=halfheadmask-halfhairmask
            facemask=image_1to3c(facemask[:,:,0])
            # etou_up_cont_pts, etou_down_cont_pts = get_cont_up_and_down(facemask, 3)
            if facemask.sum()<1000:
                continue

            etou_up_cont_pts, etou_down_cont_pts = get_etou_cont_up_and_down(facemask, landmark_in_half, meank=3)

            if etou_up_cont_pts is None:
                continue
            Calva_etoutop_pts = get_portrait_extland(etou_up_cont_pts, 20)
            draw_pts(halfheadalign, list(Calva_etoutop_pts), 10, (0, 255, 255), 5)



            draw_pts(halfheadalign, list(landmark_in_half), 5, (255, 0, 0), 5)
            draw_pts(halfheadalign, list([landmark_in_half[0],landmark_in_half[32]]), 10, (255, 0, 0), 10)


            cv2.namedWindow('halfheadalign', cv2.WINDOW_FREERATIO)
            cv2.moveWindow('halfheadalign', 0, 0)
            # cv2.imshow('halfheadalign',limit_img_auto(halfheadalign))
            cv2.imshow('halfheadalign', halfheadalign)
            limit_window_auto('halfheadalign',halfheadalign)

            get_etoupts()


            def pts2str(pts):
                pts=list(np.array(pts,np.int32))
                resstr=''
                for pt in pts:
                    resstr+=str(pt[0])+' '+str(pt[1])+','
                resstr=resstr.rstrip(',')
                return  resstr


            landmark_in_half_inori=pt_trans(landmark_in_half,param_halfhead_inori_inv)
            Calva_top_pts_inori = pt_trans(Calva_top_pts, param_halfhead_inori_inv)
            global_etou_pts_inori = pt_trans(global_etou_pts, param_halfhead_inori_inv)

            # landmark_in_half
            # Calva_etoutop_pts
            # global_etou_pts

            landstr=pts2str(landmark_in_half_inori)
            portrait_ext_str=pts2str(Calva_top_pts_inori)
            etou_str=pts2str(global_etou_pts_inori)

            all_anostr=landstr+'\n'+portrait_ext_str+'\n'+etou_str

            global_etou_pts = []
            global_val = [0, 0, 0, 0]

            print(txtpath)

            with open(txtpath,'w') as f:
                f.writelines(all_anostr)


        # if keylist[0]==13:
        #     continue
        # keylist[0] = cv2.waitKey(0)
        # if keylist[0]==27:
        #     exit(0)






