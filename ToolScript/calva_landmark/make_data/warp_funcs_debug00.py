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


def pt_trans_one(pt,param):
    x = pt[0]*param[0][0]+pt[1]*param[0][1]+param[0][2]
    y = pt[0] * param[1][0] + pt[1] * param[1][1] + param[1][2]
    return  np.array([x,y])

def euclidean(pt1,pt2):
    return  np.linalg.norm(np.array(pt1) - np.array(pt2))

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


def pts2rct(box):
    tlx = min(box[:,0])
    tly = min(box[:,1])
    brx = max(box[:,0])
    bry = max(box[:,1])
    return [tlx,tly,brx,bry]

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

def vis_delaunay(ptsnp,image):
    ptsall=list(ptsnp)
    h, w, c = image.shape
    ptsall_extend=list(ptsall)
    ptsall_extend.extend([[0, 0], [w, h]])
    rct = pts2rct(np.array(ptsall_extend))
    ptsall_np = np.array(ptsall)

    print('rct:',rct)

    offx = int(max(0, -rct[0]))
    offy = int(max(0, -rct[1]))
    nw =int( rct[2] - rct[0])
    nh =int(rct[3] - rct[1])

    ptsall_np += [offx, offy]
    bigimg = np.zeros((nh, nw, 3), dtype=image.dtype)
    bigimg[offy:offy + h, offx:offx + w] = image.copy()

    ptsnp=np.array(ptsall_np,np.int32)
    visimg=np.array(image)
    dt = Delaunay2D()
    for pt in ptsnp:
        dt.addPoint(pt)
    trires=dt.exportTriangles()
    for triind in trires:
        for k in range(0,3):
            cv2.line(bigimg, tuple(ptsnp[triind[k % 3]]), tuple(ptsnp[triind[(k +1)% 3]]), (0, 255, 0), 10)
    return bigimg

def warp_the_img(image,pt_src_list,pt_dst_list):
    h, w, c = image.shape
    ptsall_extend=list(pt_dst_list)
    ptsall_extend.extend([[0, 0], [w, h]])
    rct = pts2rct(np.array(ptsall_extend))
    offx = int(max(0, -rct[0]))
    offy = int(max(0, -rct[1]))
    nw =int( rct[2] - rct[0])
    nh =int(rct[3] - rct[1])

    ptsnp=np.array(pt_src_list,np.int32)
    visimg=np.array(image)
    dt = Delaunay2D()
    for pt in ptsnp:
        dt.addPoint(pt)
    trires=dt.exportTriangles()

    pt_src_np = np.array(pt_src_list,np.int32)+[offx, offy]
    pt_dst_np = np.array(pt_dst_list, np.int32)+[offx, offy]

    bimg_src = np.zeros((nh, nw, 3), dtype=image.dtype)
    bimg_src[offy:offy + h, offx:offx + w] = image.copy()

    bimg_dst = np.array(bimg_src)

    for triind in trires:
        triind =list(triind)

        pts_tri_src=pt_src_np[triind]
        pts_tri_dst = pt_dst_np[triind]
        if (pts_tri_src!=pts_tri_dst).any():

            ptstri_all=list(pts_tri_src)
            ptstri_all.extend(list(pts_tri_dst))
            ptstri_all=np.array(ptstri_all)
            rct_big=pts2rct(ptstri_all)
            pw=rct_big[2]-rct_big[0]
            ph=rct_big[3]-rct_big[1]
            pts_tri_src-=[rct_big[0],rct_big[1]]
            pts_tri_dst-=[rct_big[0],rct_big[1]]
            patch_src=bimg_src[rct_big[1]:rct_big[3],rct_big[0]:rct_big[2]]
            patch_dst_ori = bimg_dst[rct_big[1]:rct_big[3], rct_big[0]:rct_big[2]]
            # patch_mask_dst=np.zeros_like(patch_dst_ori )
            # cv2.fillConvexPoly(patch_mask_dst,pts_tri_dst, (1, 1, 1))
            # patch_mask_dst=patch_mask_dst.astype(np.float32)
            # cv2.fillConvexPoly(patch_src, pts_tri_src, (255, 255, 255))

            warp_param=cv2.getAffineTransform(np.array(pts_tri_src,np.float32),np.array(pts_tri_dst,np.float32))
            patch_dst=cv2.warpAffine(patch_src,warp_param,(pw,ph))
            patch_mask_dst=np.zeros_like(patch_src )
            cv2.fillConvexPoly(patch_mask_dst,pts_tri_src, (1, 1, 1))
            patch_mask_dst = cv2.warpAffine(patch_mask_dst, warp_param, (pw, ph))
            # cv2.fillConvexPoly(patch_mask_dst, pts_tri_dst, (1, 1, 1))
            # patch_mask_dst=cv2.blur(patch_mask_dst,(3,3))
            patch_mask_dst=patch_mask_dst.astype(np.float32)
            patch_dst_fusion=patch_mask_dst*patch_dst+(1-patch_mask_dst)*patch_dst_ori
            patch_dst_fusion =patch_dst_fusion.astype(np.uint8)
            bimg_dst[rct_big[1]:rct_big[3],rct_big[0]:rct_big[2]]=patch_dst_fusion.copy()
            # cv2.imshow('bimg_dst',limit_img_auto(bimg_dst))
            # key=cv2.waitKey(300)
    warp_result=bimg_dst[offy:offy + h, offx:offx + w].copy()
    return warp_result
