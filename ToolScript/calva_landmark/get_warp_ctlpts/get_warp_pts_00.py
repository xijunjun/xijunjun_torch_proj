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

# def get_mean(land98,indlist):


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

def get_ctl_pts(calva_cropedin,calva_segin,calva_matin):
    calva_croped, calva_seg, calva_mat=np.array(calva_cropedin),np.array(calva_segin),np.array(calva_matin)

    calva_matbin=img2bin_uint(calva_mat)
    # charimg=cv2.cvtColor(charimg,cv2.COLOR_BGR2GRAY)
    # contours, h = cv2.findContours(calva_matbin.copy()[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, h = cv2.findContours(calva_matbin.copy()[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    calva_croped = cv2.drawContours(calva_croped, [contours[0]], -1, (0, 255, 255), 10)

    cont_ptlist=[]
    for pt in contours[0]:
        cont_ptlist.append(pt[0])
        # print(pt)
    for pt in cont_ptlist:
        cv2.circle(calva_croped, tuple(pt), 20, (255, 255, 255), thickness=-1)

    cont_ptlist=np.array(cont_ptlist)

    numpts = cont_ptlist.shape[0]
    # cont_ptlist=cont_ptlist[::numpts//20]
    # cont_ptlist[-1] = cont_ptlist[0]

    numpts=cont_ptlist.shape[0]
    # csX = CubicSpline(np.arange(numpts), cont_ptlist[:,0], bc_type='periodic')
    # csY = CubicSpline(np.arange(numpts), cont_ptlist[:,1], bc_type='periodic')
    # N=numpts
    # IN=np.linspace(0, N - 1, 1 * N)
    # curvex=csX(IN)
    # curvey=csY(IN)
    # for i in range(0,N):
    #     cv2.circle(calva_croped, (int(curvex[i]), int(curvey[i])), 10, (255, 0, 255), thickness=-1)

    # contlist_pd=Periodic_list(listin=cont_ptlist)
    # for pt in contlist_pd:
    #     print(pt)

    contlist_pd=smo_the_pts_close(cont_ptlist,200)
    # contlist_pd = smo_the_pts_close(contlist_pd, 400)

    for i in range(0,numpts):
        cv2.circle(calva_croped, (int(contlist_pd[i][0]), int(contlist_pd[i][1])), 10, (255, 0, 255), thickness=-1)
        # cv2.imshow('calva_cropedbig', limit_img_auto(calva_croped))
        # key = cv2.waitKey(1)
        # if key==27:
        #     break

    # for i,cur_color in enumerate(part_colors):
    #     hairseg = get_target_seg(calva_seg,cur_color)
    #     cv2.putText(hairseg , str(i), (400, 400), cv2.FONT_HERSHEY_DUPLEX, 4,(255, 255, 255))
    #     cv2.imshow('hairseg', limit_img_auto(hairseg))
    #     cv2.waitKey(0)

    hairseg = get_target_seg(calva_seg,part_colors[17])
    # cv2.putText(hairseg , str(i), (400, 400), cv2.FONT_HERSHEY_DUPLEX, 4,(255, 255, 255))
    # cv2.imshow('hairseg', limit_img_auto(hairseg))
    #
    cv2.imshow('calva_cropedbig',limit_img_auto(calva_croped))

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

    # draw_pts(calva_mat_new, list([ptlist[lind], ptlist[rind]]), 20, (255, 0, 0), 2)
    # draw_pts(calva_mat_new,ptlist,2,(255,0,0),2)
    # draw_pts(calva_mat_new,list(sublist1),2,(255,0,255),2)
    # draw_pts(calva_mat_new, list(sublist2), 2, (0, 255, 0), 2)
    # cv2.imshow('calva_mat_new',limit_img_auto(calva_mat_new))

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


def expand_the_pts(exp_base_pts, exp_pts):
    base_pt = exp_base_pts[0]
    exp_pt_result = []
    exp_ratio=1.15
    shape_param=[0.3,0.7,0.9,1.0]
    # shape_param = [0.3, 0.6, 0.8, 1.0]

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


def merge_pts(ptslist_list):
    res = []
    for ptlist in list(ptslist_list):
        res.extend(ptlist)
    return list(res)

def make_backgroud_stable_pts(expand_pts,bt_pts,exp_base_pts):
    base_pt=exp_base_pts[0]
    back_stable_pts=[]
    ptsall=merge_pts([expand_pts,bt_pts])
    stable_pt_list=[]
    for pt in ptsall:
        lenth=euclidean(pt,base_pt)
        xpt=[lenth,0]
        trans_param=cv2.estimateAffinePartial2D(np.array([base_pt,pt]),np.array([[0,0],xpt]), method=cv2.LMEDS)[0]
        trans_param_inv=cv2.invertAffineTransform(trans_param)
        xpt=pt_trans_one([lenth*2,0],trans_param_inv)
        stable_pt_list.append(np.array(xpt,np.int32))
    return stable_pt_list




def make_big_img(ptslist_list, image):
    ptsall = merge_pts(ptslist_list)
    h, w, c = image.shape
    ptsall.extend([[0, 0], [w, h]])
    ptsall_np = np.array(ptsall)
    rct = pts2rct(ptsall_np)

    offx = max(0, -rct[0])
    offy = max(0, -rct[1])
    nw = rct[2] - rct[0]
    nh = rct[3] - rct[1]

    ptsall_np += [offx, offy]
    bigimg = np.zeros((nh, nw, 3), dtype=image.dtype)
    bigimg[offy:offy + h, offx:offx + w] = image.copy()
    draw_pts(bigimg, list(ptsall_np), 10, (0, 255, 255), 10)
    return bigimg




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


if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)
    matnet = init_matting_model()
    bise_net = init_parsing_model(model_name='bisenet')

    # srcroot = '/home/tao/mynas/Dataset/FaceEdit/sumiao/'
    # srcroot=r'/home/tao/mynas/Dataset/hairforsr/femalehd'
    srcroot=r'/home/tao/Downloads/image_unsplash'
    dstroot = '/home/tao/disk1/Dataset/Project/FaceEdit/taobao_sumiao/crop/'

    # srcroot='/home/tao/mynas/Dataset/FaceEdit/sumiao'

    ims = get_ims(srcroot)
    # face_size = 2048
    head_size = 2536


    for i, im in enumerate(ims):
        all_face_rcts = []
        all_face_lands = []
        frame = cv2.imread(im)
        image_const=np.array(frame)

        bbox_list, landmark_list = get_face_land_rct(det_net, align_net, image_const)

        for j,landmarks in enumerate(landmark_list):

            ##########裁剪出单张人头
            land5_from98=land98to5(landmarks)
            warp_param_head,warp_param_face_inv=get_crop_param(land5_from98)
            headalign = cv2.warpAffine(image_const, warp_param_head, (head_size, head_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
            h,w,c=headalign.shape

            # print('facealign.shape ',facealign.shape)

            ##########获取单张人头的matting和seg结果
            head_mat = get_mat(matnet, headalign)
            head_seg_bise = pred_seg_bise(bise_net, headalign)
            head_mat_3c = image_1to3c(head_mat)

            ##########获取单张人头的关键点
            land98_in_crop=pt_trans(landmarks,warp_param_head)
            for pt in land98_in_crop:
                pt=np.array(pt,np.int32)
                cv2.circle(headalign, (pt[0], pt[1]), 10, (255, 0, 0), -1, -1)

            headlandmask = get_landmask_mask(headalign, land98_in_crop)

            ##########获取颅顶裁剪框
            calva_bottom_y=int(get_calva_bottom(land98_in_crop))
            # print(warp_param_face_2048)
            calva_limit_rct,calva_crop_rct=get_calva_crop_rct(headalign, head_mat_3c, land98_in_crop)
            # cv2.rectangle(facealign, (calva_limit_rct[0], calva_limit_rct[1]), (calva_limit_rct[2], calva_limit_rct[3]), (0, 255, 255), 10)
            # cv2.rectangle(facealign, (calva_crop_rct[0], calva_crop_rct[1]), (calva_crop_rct[2], calva_crop_rct[3]), (0, 0, 255), 10)

            ##########从大图中裁剪颅顶
            calva_crop_quad=[[calva_crop_rct[0],calva_crop_rct[1]],[calva_crop_rct[2],calva_crop_rct[1]],[calva_crop_rct[2],calva_crop_rct[3]],[calva_crop_rct[0],calva_crop_rct[3]]]
            calva_quad_inv=pt_trans(list(calva_crop_quad) ,warp_param_face_inv)
            calva_quad_inv=np.array(calva_quad_inv,np.int32)
            for k in range(0,4):
                # cv2.line(facealign, tuple(calva_crop_quad[k % 4]), tuple(calva_crop_quad[(k + 1) % 4]), (0, 255, 0), 10)
                cv2.line(frame, tuple(calva_quad_inv[k % 4]), tuple(calva_quad_inv[(k + 1) % 4]), (0, 255, 0), 10)
            calva_dst_quad=np.array([[0,0],[1536,0],[1536,1280],[0,1280]])
            cava_crop_param = cv2.estimateAffinePartial2D(calva_quad_inv, calva_dst_quad, method=cv2.LMEDS)[0]
            cava_crop_param_inv = cv2.invertAffineTransform(cava_crop_param)

            calva_croped = cv2.warpAffine(image_const, cava_crop_param, (1536, 1280), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))

            ################################颅顶在人脸图中的位置
            calva_in_align_param=cv2.estimateAffinePartial2D(np.array(calva_crop_quad), calva_dst_quad, method=cv2.LMEDS)[0]

            # calva_mat=cv2.warpAffine(face_mat_3c, cava_crop_param, (1536, 1280), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            # cv2.imshow('seg_bise', limit_img_auto(np.concatenate([facealign,face_mat_3c, matbin,matbin_sim], axis=1)))
            # cropimg, xshift, yshift=crop_pad(facealign,croprct.reshape(2,2))

            ######### warp #################
            # calva_seg=pred_seg_bise(bise_net, calva_croped)
            # calva_mat =image_1to3c(get_mat(matnet,calva_croped))
            calva_seg = cv2.warpAffine(head_seg_bise, calva_in_align_param, (1536, 1280), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),flags=cv2.INTER_NEAREST)
            calva_mat = cv2.warpAffine(head_mat_3c, calva_in_align_param, (1536, 1280), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),flags=cv2.INTER_NEAREST)
            calvalandmask=cv2.warpAffine(headlandmask, calva_in_align_param, (1536, 1280), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))



            ##############构造控制点
            # get_ctl_pts(small_to_big(calva_croped), small_to_big(calva_seg), small_to_big(calva_mat))
            # get_ctl_pts(calva_croped, calva_seg, calva_mat)
            up_cont_pts,down_cont_pts=get_cont_up_and_down(calva_mat)

            ##底部两个关键点
            calva_mat_bin = img2bin_uint(calva_mat)
            ch,cw,cc=calva_mat.shape
            rct = cv2.boundingRect(calva_mat_bin[:, :, 0])
            pt_bl = [rct[0], ch]
            pt_br = [rct[0] + rct[2], ch]
            Calva_bottom_pts = [pt_bl, pt_br]


            ##扩张基准点
            calva_land = get_calva_landmark(align_net, calva_croped)
            Calva_base_pts=[calva_land[51]]

            ####构造扩张点
            Calva_expand_pts=get_expand_pts(Calva_base_pts, up_cont_pts, Calva_bottom_pts)
            # Calva_expand_pts=get_expand_pts_divy(up_cont_pts, calva_mat_bin)

            Calva_expand_pts_result = expand_the_pts(Calva_base_pts, Calva_expand_pts)


            hairmask=get_target_seg(calva_seg,part_colors[17])
            headmask=255-get_target_seg(calva_seg,[255,255,255])
            facemask=headmask-hairmask
            facemask=image_1to3c(facemask[:,:,0])
            print(facemask.shape,calvalandmask.shape)

            # landmask = get_landmask_mask(calva_croped, land98_in_crop)
            # facemask=np.clip(facemask+calvalandmask,0,255)
            facemask[calvalandmask>0]=255

            # facemask_forell=
            facemask_upflip = cv2.flip(facemask, 0)
            facemask_forell=np.concatenate([facemask,facemask_upflip],axis=0)

            contours, h = cv2.findContours(facemask_forell[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            minEllipse = cv2.fitEllipse(contours[0])
            cv2.ellipse(facemask_forell, minEllipse, (255, 255, 255), -1)
            facemask_adde=facemask_forell[0:1280,0:1536]
            facemask_adde=cv2.dilate(facemask_adde,(19,19),iterations=4)

            facemask_up_cont_pts, facemask_down_cont_pts = get_cont_up_and_down(facemask_adde,meank=3)
            Calva_etou_pts = get_expand_pts(Calva_base_pts, facemask_up_cont_pts, Calva_bottom_pts)

            draw_pts(facemask_adde, list(Calva_etou_pts), 20, (255, 0, 0), 2)

            # cv2.imshow('facemask_forell', limit_img_auto(facemask_forell))
            cv2.imshow('facemask_adde',limit_img_auto(facemask_adde))




            cv2.imshow('facemask',limit_img_auto(facemask))
            cv2.imshow('landmask',limit_img_auto(calvalandmask))
            cv2.imshow('headalign',limit_img_auto(headalign))
            print(headalign.shape)




            Calva_back_stable_pts=make_backgroud_stable_pts(Calva_expand_pts_result,Calva_bottom_pts,Calva_base_pts)
            bigimg=make_big_img([Calva_back_stable_pts,Calva_expand_pts],calva_croped)

            print(Calva_bottom_pts)
            calva_croped_vis=vis_delaunay(merge_pts([Calva_base_pts,Calva_expand_pts,Calva_bottom_pts,Calva_back_stable_pts,Calva_etou_pts]), calva_croped)


            # pt_src_list=merge_pts([Calva_bottom_pts,Calva_base_pts,Calva_expand_pts,Calva_back_stable_pts,Calva_etou_pts])
            # pt_dst_list=merge_pts([Calva_bottom_pts,Calva_base_pts,Calva_expand_pts_result,Calva_back_stable_pts,Calva_etou_pts])
            # warp_result=warp_the_img(calva_croped, pt_src_list, pt_dst_list)

            pt_src_list=merge_pts([Calva_bottom_pts,Calva_base_pts,Calva_expand_pts,Calva_back_stable_pts,Calva_etou_pts])
            pt_dst_list=merge_pts([Calva_bottom_pts,Calva_base_pts,Calva_expand_pts_result,Calva_back_stable_pts,Calva_etou_pts])

            pt_src_list_inv=pt_trans(list(pt_src_list) ,cava_crop_param_inv)
            pt_src_list_inv=np.array(pt_src_list_inv,np.int32)
            pt_dst_list_inv = pt_trans(list(pt_dst_list), cava_crop_param_inv)
            pt_dst_list_inv = np.array(pt_dst_list_inv, np.int32)


            warp_result=warp_the_img(image_const, pt_src_list_inv, pt_dst_list_inv)


            cv2.imshow('calva_croped_vis',limit_img_auto(calva_croped_vis))
            cv2.imshow('warp_result',limit_img_auto(warp_result))




            draw_pts(calva_mat, list(up_cont_pts), 20, (255, 0, 0), 2)
            draw_pts(calva_mat, list(down_cont_pts), 20, (255, 255, 0), 2)
            draw_pts(calva_mat, list(Calva_bottom_pts), 30, (0, 0, 255), 30)
            draw_pts(calva_mat, list(Calva_base_pts), 30, (0, 255, 255), 30)
            # draw_pts(calva_croped, list(calva_land), 10, (0, 255, 255), 10)
            draw_pts(calva_seg, list(Calva_base_pts), 30, (0, 255, 255), 30)
            draw_pts(calva_mat, list(Calva_expand_pts), 30, (0, 255, 255), 30)
            draw_pts(calva_mat, list(Calva_expand_pts_result), 30, (0, 255, 255), 30)




        cv2.imshow('bimg',limit_img_auto(bigimg))
        cv2.imshow('cat',limit_img_auto(np.concatenate([calva_croped,calva_seg,calva_mat],axis=1)))

        show_img_two(image_const, warp_result)

        # landmarks = align_net.get_landmarks(frame)
        # landmarks = landmarks.astype(np.int32)
        # cv2.imwrite(dstroot+os.path.basename(im),facealign)
        # cv2.imshow("capture", limit_img_auto(frame))
        # cv2.imshow("facealign", limit_img_auto(facealign))
        # cv2.imshow('calva_croped',limit_img_auto(calva_croped))

        # if cv2.waitKey(10) & 0xff == ord('q'):
        #     break

        if keylist[0]==13:
            continue
        keylist[0] = cv2.waitKey(0)
        if keylist[0]==27:
            exit(0)
