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

def euclidean(pt1,pt2):
    return  np.linalg.norm(np.array(pt1) - np.array(pt2))

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

def img2bin_uint(imgin):
    img=np.array(imgin)
    thres=100
    img[img<thres]=0
    img[img>=thres]=255
    return   img

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


def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0

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


def min_dis_ind(pt1,ptlist):
    dis_list=[]
    for pt in ptlist:
        dis_list.append(euclidean(pt,pt1))
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
    calva_mat_new=np.array(calva_mat)
    calva_mat_bin=img2bin_uint(calva_mat)

    contours, _ = cv2.findContours(calva_mat_bin[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    ptlist=[]
    for pt in contours[0]:
        ptlist.append(pt[0])
    ptlist=smo_the_pts(ptlist,meank)
    ptlist = smo_the_pts(ptlist, meank)

    cornet_lb=[0,h]
    corner_rb=[w,h]
    lind=min_dis_ind(cornet_lb, ptlist)
    rind = min_dis_ind(corner_rb, ptlist)
    sublist1, sublist2 = split_cont_by_two_pts(ptlist, ptlist[lind], ptlist[rind])

    return list(sublist1),list(sublist2)