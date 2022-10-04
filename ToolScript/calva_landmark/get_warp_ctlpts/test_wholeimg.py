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


from warpfuncs_xlib import *
from warpfuncs_triwarp import *
from warpfuncs_getpts  import *


def limit_img_auto(imgin):
    img=np.array(imgin)
    sw = 1920 * 1.2
    sh = 1080 * 1.2
    h, w = tuple(list(imgin.shape)[0:2])
    swhratio = 1.0 * sw / sh
    whratio = 1.0 * w / h
    resize_ratio=sh/h
    if whratio > swhratio:
        resize_ratio=1.0*sw/w
    if resize_ratio<1:
        img=cv2.resize(imgin,None,fx=resize_ratio,fy=resize_ratio)
    return img


def get_ims(imgpath):
    imgpathlst = []
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def draw_pts(img,ptlist,r,color,thick,wait=0):
    for pt in ptlist:
        cv2.circle(img,tuple(np.array(pt,np.int32)),r,color,thick)


def get_standhead_crop_param_targetsize(landpts5,targetsize):
    template_2048=np.array([[863,1147],[1217,1147],[1043,1383],[889,1547],[1193,1547]])
    template_2048 +=244
    template_nrom=template_2048/2536
    head_temp=template_nrom*targetsize
    warp_param_face_2048=cv2.estimateAffinePartial2D(landpts5, head_temp, method=cv2.LMEDS)[0]
    warp_param_face_inv=cv2.invertAffineTransform(warp_param_face_2048)

    return warp_param_face_2048,warp_param_face_inv

def get_etou_crop_rct_byland(landmarks,headw):
    h, w, c = headw,headw,3
    landmarksnp=np.array(landmarks,np.int32)

    stand_etouw=1280
    stand_etouh=960
    stand_etouwh_ratio=1.0*stand_etouw/stand_etouh

    wextend_ratio=1.8
    hextend_ratio = 1.8
    bry=int(landmarks[54][1])
    minlandx =landmarksnp[:,0] [np.argmin(landmarksnp[:,0])]
    maxlandx =landmarksnp[:,0] [np.argmax(landmarksnp[:, 0])]
    minlandy = landmarksnp[:,1] [np.argmin(landmarksnp[:,1])]
    land_w=maxlandx-minlandx
    land_h=bry-minlandy

    extend_w = land_w * wextend_ratio
    extend_h = land_h * hextend_ratio

    ctx=(maxlandx+minlandx)/2
    # cty = land_h / 2
    extended_tlx = ctx-extend_w/2
    extended_brx = extended_tlx + extend_w
    extended_tly=bry-extend_h

    cur_wh_ratio=1.0*extend_w /extend_h
    #哪边小就扩哪边
    if cur_wh_ratio<stand_etouwh_ratio:
        deltaw=(stand_etouwh_ratio-cur_wh_ratio)*extend_h
        extended_tlx -= deltaw/2
        extended_brx += deltaw/2
    else:
        deltah = (1.0/stand_etouwh_ratio-1.0/cur_wh_ratio)*extend_w
        extended_tly-=deltah
    etou_rct=[extended_tlx,extended_tly,extended_brx,bry]
    etou_rct=list(np.array(etou_rct,np.int32))

    return etou_rct,stand_etouh,stand_etouw


def img2tensor(img):
    img = img.transpose(2, 0, 1)
    imgtensor = torch.from_numpy(img)
    imgtensor=imgtensor.unsqueeze(0)
    return imgtensor

def pred_etou_land(etou_net,imgin):
    device='cpu'

    img=np.array(imgin,np.float32) / 255.0
    img=cv2.resize(img,(112,112))

    # cv2.imshow('img11',img/255.0)

    land_pred = etou_net(img2tensor(img).to(device))
    land_pred = land_pred.cpu().numpy()
    etouland = land_pred * 1280
    return np.array(etouland,np.float32)


if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)
    matnet = init_matting_model()
    bise_net = init_parsing_model(model_name='bisenet')

    # srcroot = '/home/tao/mynas/Dataset/FaceEdit/sumiao/'
    # srcroot=r'/home/tao/mynas/Dataset/hairforsr/femalehd'
    srcroot=r'/home/tao/Downloads/image_unsplash'
    # srcroot='/home/tao/mynas/Dataset/FaceEdit/sumiao'
    dstroot = '/home/tao/disk1/Workspace/TrainResult/eland/testim2/'

    ims = get_ims(srcroot)
    head_size = 2048

    etou_net=torch.jit.load('/home/tao/disk1/Workspace/TrainResult/eland/eland01/plate_land_latest_jit.pt').to('cpu')



    for i, im in enumerate(ims):
        imname=os.path.basename(im)

        all_face_rcts = []
        all_face_lands = []
        frame = cv2.imread(im)
        image_const=np.array(frame)

        bbox_list, landmark_list = get_face_land_rct(det_net, align_net, image_const)

        img=cv2.imread(im)
        imgvis=np.array(img)

        for j,faceland_ori in  enumerate(landmark_list):
            draw_pts(imgvis, list(faceland_ori), 10, (0, 255, 0), 10)

            ##########裁剪出单张人头
            land5_from98=land98to5(faceland_ori)
            wparam_ori_to_standhead,wparam_ori_to_standhead_inv=get_standhead_crop_param_targetsize(land5_from98,2048)
            img_standhead= cv2.warpAffine(image_const, wparam_ori_to_standhead, (head_size, head_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
            h,w,c=img_standhead.shape

            # print('facealign.shape ',facealign.shape)

            # ##########获取单张人头的matting和seg结果
            # head_mat = get_mat(matnet, headalign)
            # head_seg_bise = pred_seg_bise(bise_net, headalign)
            # head_mat_3c = image_1to3c(head_mat)

            ##########获取单张人头的关键点
            faceland_standhead=pt_trans(faceland_ori,wparam_ori_to_standhead)



            #############etou############################
            etou_crop_rct, etouh, etouw = get_etou_crop_rct_byland(faceland_standhead, 2048)
            etou_quad_incrop = [[etou_crop_rct[0], etou_crop_rct[1]], [etou_crop_rct[2], etou_crop_rct[1]],
                                [etou_crop_rct[2], etou_crop_rct[3]], [etou_crop_rct[0], etou_crop_rct[3]]]
            etou_quad_incrop = np.array(etou_quad_incrop, np.float32)
            etou_quad_inv = pt_trans(list(etou_quad_incrop), wparam_ori_to_standhead_inv)
            etou_quad_inv = np.array(etou_quad_inv, np.int32)
            etoucroped_dst_quad = np.array([[0, 0], [etouw, 0], [etouw, etouh], [0, etouh]])
            wparam_ori_to_etou = cv2.estimateAffinePartial2D(etou_quad_inv, etoucroped_dst_quad, method=cv2.LMEDS)[0]
            wparam_ori_to_etou_inv = cv2.invertAffineTransform(wparam_ori_to_etou)
            img_etou_croped=cv2.warpAffine(img, wparam_ori_to_etou, (etouw, etouh), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))

            cv2.imshow('img_etou_croped',limit_img_auto(img_etou_croped))
            cv2.imwrite(os.path.join(dstroot,imname),img_etou_croped)



            etou_land=pred_etou_land(etou_net,img_etou_croped)
            print(etou_land)
            etou_land=etou_land.reshape(-1, 2)
            print(etou_land)

            etou_land_inori=pt_trans(list(etou_land),wparam_ori_to_etou_inv)

            print(etou_land_inori)

            # etou_land_croped = pt_trans(etou_land, wparam_ori_to_etou)
            draw_pts(imgvis, list(etou_land_inori), 10, (0, 0, 255), 10)


        cv2.imshow('img',limit_img_auto(imgvis))

        if cv2.waitKey(0)==27:
            exit(0)