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


def get_crop_param_targetsize(landpts5,targetsize):
    template_2048=np.array([[863,1147],[1217,1147],[1043,1383],[889,1547],[1193,1547]])
    template_2048 +=244
    template_nrom=template_2048/2536
    head_temp=template_nrom*targetsize

    warp_param_face_2048=cv2.estimateAffinePartial2D(landpts5, head_temp, method=cv2.LMEDS)[0]
    warp_param_face_inv=cv2.invertAffineTransform(warp_param_face_2048)

    return warp_param_face_2048,warp_param_face_inv

def pt_trans(pts,param):

    dst=[]
    for pt in pts:
        x = pt[0]*param[0][0]+pt[1]*param[0][1]+param[0][2]
        y = pt[0] * param[1][0] + pt[1] * param[1][1] + param[1][2]
        dst.append([x,y])
    return  np.array(dst)

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


def get_etou_crop_rct(landmarks,headw):
    h, w, c = headw,headw,3
    etouw=1280
    # etouh=int(etouw*10/16)
    etouh = int(etouw * 12 / 16)

    limit_whratio=1.0*etouw/etouh
    calva_bottom_y=int(landmarks[54][1])
    upy=calva_bottom_y-etouh
    ctx=headw//2
    print('etouw,etouh:',etouw,etouh)
    etou_rct = [ctx-etouw//2, upy]
    etou_rct = [etou_rct[0], etou_rct[1], etou_rct[0]+etouw, upy+etouh]

    # half_head_rct = [0, upy, headw, calva_bottom_y]
    etou_rct=np.array(etou_rct,np.int32)
    half_headh, half_headw=etouh,etouw

    return etou_rct,half_headh,half_headw


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

    print('extend_w /extend_h',extend_w ,extend_h)
    print('stand_etouwh_ratio,cur_wh_ratio:',stand_etouwh_ratio,cur_wh_ratio)

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


def get_valid_impaths(imroot):

    ims=get_ims(imroot)
    validims=[]
    for i, im in enumerate(ims):
        imkey, ext=get_imkey_ext(im)
        txtpath=os.path.join(imroot,imkey+'_calvaland.txt')
        if os.path.exists(txtpath) is False:
            continue
        faceland,portait_ext_land,etou_land=load_calva_land(txtpath)
        if len(list(etou_land))!=20:
            print(len(list(etou_land)),im)
            continue
        validims.append((im,txtpath))
    return validims


if __name__=='__main__':
    srcroot=r'/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/ffhq'
    dstroot='/home/tao/disk1/Workspace/TrainResult/eland/testim'

    # validims=get_valid_impaths(srcroot)
    # for i,tup in enumerate(validims):
    #     im, txtpath=tup
    #     print(im, txtpath)
    #
    #
    # exit(0)


    ims = get_ims(srcroot)

    head_size=2048

    for i, im in enumerate(ims):
        imkey, ext=get_imkey_ext(im)
        print(imkey, ext)
        txtpath=os.path.join(srcroot,imkey+'_calvaland.txt')

        # if imkey!='FFHQ_01487':
        #     continue

        if os.path.exists(txtpath) is False:
            continue

        faceland,portait_ext_land,etou_land=load_calva_land(txtpath)
        img=cv2.imread(im)
        img_visual=np.array(img)

        draw_pts(img_visual, list(faceland), 10, (0, 255, 0), 5)
        # draw_pts(img, list(portait_ext_land), 10, (255, 255, 0), 5)
        draw_pts(img_visual, list(etou_land), 20, (0, 255, 255), 5)


        land5_from98 = land98to5(faceland)
        warp_param_head, warp_param_face_inv = get_crop_param_targetsize(land5_from98, head_size)
        headalign = cv2.warpAffine(img, warp_param_head, (head_size, head_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))

        faceland_in_crop=pt_trans(faceland,warp_param_head)
        etou_land_in_crop=pt_trans(etou_land,warp_param_head)

        # etou_crop_rct,etouh,etouw=get_etou_crop_rct(faceland_in_crop, 2048)
        etou_crop_rct, etouh, etouw = get_etou_crop_rct_byland(faceland_in_crop, 2048)



        etou_quad_incrop = [[etou_crop_rct[0], etou_crop_rct[1]], [etou_crop_rct[2], etou_crop_rct[1]],
                                [etou_crop_rct[2], etou_crop_rct[3]], [etou_crop_rct[0], etou_crop_rct[3]]]
        etou_quad_incrop = np.array(etou_quad_incrop, np.float32)

        etou_quad_inv = pt_trans(list(etou_quad_incrop), warp_param_face_inv)
        etou_quad_inv = np.array(etou_quad_inv, np.int32)


        etoucroped_dst_quad = np.array([[0, 0], [etouw, 0], [etouw, etouh], [0, etouh]])
        param_etoucroped_inori = cv2.estimateAffinePartial2D(etou_quad_inv, etoucroped_dst_quad, method=cv2.LMEDS)[0]
        param_etoucroped_inori_inv=cv2.invertAffineTransform(param_etoucroped_inori)
        etou_land_croped=pt_trans(etou_land,param_etoucroped_inori)

        etou_croped= cv2.warpAffine(img, param_etoucroped_inori, (etouw, etouh), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))


        # draw_pts(etou_croped, list(etou_land_croped), 10, (0, 255, 255), 5)
        #
        # etou_croped=cv2.resize(etou_croped,(128,96))
        # etou_croped = cv2.resize(etou_croped, (etouw, etouh),cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(dstroot,os.path.basename(im)),etou_croped)


        cv2.imshow('etou_croped',limit_img_auto(etou_croped))

        draw_pts(headalign,list(etou_quad_incrop ), 30, (0, 255, 0), 5)
        cv2.rectangle(headalign,(etou_crop_rct[0],etou_crop_rct[1]),(etou_crop_rct[2],etou_crop_rct[3]),color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)






        cv2.imshow('img_visual',limit_img_auto(img_visual))
        cv2.imshow('headalign ',limit_img_auto(headalign ))


        if cv2.waitKey(0)==27:
            exit(0)










