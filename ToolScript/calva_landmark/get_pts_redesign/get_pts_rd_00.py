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

    # wextend_ratio=1.8
    wextend_ratio = 1.8
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


def get_halfhead_crop_rct(landmarks,headw):

    h, w, c = headw,headw,3
    calvaw=headw
    calvah=int(headw*10/16)
    limit_whratio=1.0*calvaw/calvah
    calva_bottom_y=int(landmarks[54][1])
    upy=calva_bottom_y-calvah
    # half_head_rct=[[0,upy],[headw,calva_bottom_y]]
    half_head_rct = [0, upy, headw, calva_bottom_y]
    half_head_rct=np.array(half_head_rct,np.int32)
    half_headh, half_headw=calvah,calvaw

    return half_head_rct,half_headh,half_headw


def img2tensor(img):
    img = img.transpose(2, 0, 1)
    imgtensor = torch.from_numpy(img)
    imgtensor=imgtensor.unsqueeze(0)
    return imgtensor

def pred_etou_land(etou_net,imgin):
    device='cpu'
    # insize=224
    insize = 112

    img=np.array(imgin,np.float32) / 255.0
    img=cv2.resize(img,(insize,insize))

    # cv2.imshow('img11',img/255.0)

    land_pred = etou_net(img2tensor(img).to(device))
    land_pred = land_pred.cpu().numpy()
    etouland = land_pred * 1280
    return np.array(etouland,np.float32)

def get_etou_ctlpts(img,faceland_ori,etou_net):
    land5_from98 = land98to5(faceland_ori)
    standheadsize=2048
    wparam_ori_to_standhead, wparam_ori_to_standhead_inv = get_standhead_crop_param_targetsize(land5_from98, standheadsize)

    faceland_standhead = pt_trans(faceland_ori, wparam_ori_to_standhead)
    #############etou############################
    etou_crop_rct, etouh, etouw = get_etou_crop_rct_byland(faceland_standhead, standheadsize)
    etou_quad_incrop = [[etou_crop_rct[0], etou_crop_rct[1]], [etou_crop_rct[2], etou_crop_rct[1]],
                        [etou_crop_rct[2], etou_crop_rct[3]], [etou_crop_rct[0], etou_crop_rct[3]]]
    etou_quad_incrop = np.array(etou_quad_incrop, np.float32)
    etou_quad_inv = pt_trans(list(etou_quad_incrop), wparam_ori_to_standhead_inv)
    etou_quad_inv = np.array(etou_quad_inv, np.int32)
    etoucroped_dst_quad = np.array([[0, 0], [etouw, 0], [etouw, etouh], [0, etouh]])
    wparam_ori_to_etou = cv2.estimateAffinePartial2D(etou_quad_inv, etoucroped_dst_quad, method=cv2.LMEDS)[0]
    wparam_ori_to_etou_inv = cv2.invertAffineTransform(wparam_ori_to_etou)
    img_etou_croped = cv2.warpAffine(img, wparam_ori_to_etou, (etouw, etouh), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
    etou_land = pred_etou_land(etou_net, img_etou_croped)
    etou_land = etou_land.reshape(-1, 2)
    etou_land=interpolate_pts(etouw, etouh, list(etou_land))
    etou_land_inori = pt_trans(list(etou_land), wparam_ori_to_etou_inv)



    standctx=standheadsize//2
    standcty = standheadsize // 2
    # backstablesize=int(standheadsize//2*3)
    backstablesize =int(standheadsize/3*2)

    backstable_quad_instand = [[standctx-backstablesize,standcty-backstablesize],[standctx,standcty-backstablesize],[standctx + backstablesize, standcty - backstablesize],
                               [standctx + backstablesize, standcty ],[standctx-backstablesize,standcty],
                               [standctx - backstablesize, standcty + backstablesize],[standctx+backstablesize,standcty+backstablesize]
                               ]
    print(backstable_quad_instand)
    backstable_quad_instand_inv=pt_trans(list(backstable_quad_instand), wparam_ori_to_standhead_inv)

    return etou_land_inori,backstable_quad_instand_inv


def get_headout_ctlpts(image,faceland_ori,matnet,bise_net):
    head_size=2048
    land5_from98 = land98to5(faceland_ori)
    wparam_ori_to_standhead, wparam_ori_to_standhead_inv = get_standhead_crop_param_targetsize(land5_from98, 2048)
    faceland_standhead = pt_trans(faceland_ori, wparam_ori_to_standhead)

    headalign = cv2.warpAffine(image, wparam_ori_to_standhead, (head_size, head_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
    head_mat = get_mat(matnet, headalign)
    head_seg_bise = pred_seg_bise(bise_net, headalign)
    head_mat_3c = image_1to3c(head_mat)

    halfhead_crop_rct, half_headh, half_headw = get_halfhead_crop_rct(faceland_standhead, head_size)
    halfhead_quad_incrop = [[halfhead_crop_rct[0], halfhead_crop_rct[1]], [halfhead_crop_rct[2], halfhead_crop_rct[1]],
                            [halfhead_crop_rct[2], halfhead_crop_rct[3]], [halfhead_crop_rct[0], halfhead_crop_rct[3]]]
    halfhead_quad_incrop = np.array(halfhead_quad_incrop, np.float32)
    halfhead_quad_inv = pt_trans(list(halfhead_quad_incrop), wparam_ori_to_standhead_inv)
    halfhead_quad_inv = np.array(halfhead_quad_inv, np.int32)

    halfhead_dst_quad = np.array([[0, 0], [half_headw, 0], [half_headw, half_headh], [0, half_headh]])
    param_halfhead_incrop = cv2.estimateAffinePartial2D(halfhead_quad_incrop, halfhead_dst_quad, method=cv2.LMEDS)[0]

    halfheadalign = cv2.warpAffine(headalign, param_halfhead_incrop, (half_headw, half_headh), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
    halfhead_mat_3c = cv2.warpAffine(head_mat_3c, param_halfhead_incrop, (half_headw, half_headh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    halfhead_seg_bise = cv2.warpAffine(head_seg_bise, param_halfhead_incrop, (half_headw, half_headh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    wparam_ori_to_halfhead = cv2.estimateAffinePartial2D(halfhead_quad_inv, halfhead_dst_quad, method=cv2.LMEDS)[0]
    wparam_ori_to_halfhead_inv = cv2.invertAffineTransform(wparam_ori_to_halfhead)
    landmark_in_halfhead = pt_trans(faceland_ori, wparam_ori_to_halfhead )


    calva_mat_bin = img2bin_uint(halfhead_mat_3c)

    calva_mat_bin = cv2.erode(calva_mat_bin, kernel=np.ones((19, 19), np.uint8), iterations=2)
    calva_mat_bin = cv2.dilate(calva_mat_bin, kernel=np.ones((19, 19), np.uint8), iterations=2)
    calva_mat_bin = cv2.dilate(calva_mat_bin, kernel=np.ones((9, 9), np.uint8), iterations=2)
    calva_mat_bin = cv2.erode(calva_mat_bin, kernel=np.ones((9, 9), np.uint8), iterations=2)

    up_cont_pts, down_cont_pts ,btpts= get_cont_up_and_down(calva_mat_bin, 3)

    ch, cw, cc = calva_mat_bin.shape
    rct = cv2.boundingRect(calva_mat_bin[:, :, 0])
    pt_bl = [rct[0], ch]
    pt_br = [rct[0] + rct[2], ch]
    Calva_bottom_pts = [pt_bl, pt_br]

    ##扩张基准点
    Calva_base_pts = [landmark_in_halfhead[51]]
    ####构造扩张点
    Calva_expand_pts = get_expand_pts(Calva_base_pts, up_cont_pts, Calva_bottom_pts)
    # Calva_expand_pts_result = expand_the_pts(Calva_base_pts, Calva_expand_pts)

    ctlpts=merge_pts([Calva_base_pts,btpts,Calva_expand_pts])
    ctlpts_inv=pt_trans(ctlpts,wparam_ori_to_halfhead_inv)

    cv2.imshow('calva_mat_bin',limit_img_auto(calva_mat_bin))

    Calva_expand_pts_inori=pt_trans(Calva_expand_pts ,wparam_ori_to_halfhead_inv)
    Calva_stable_hairbt_pts_inori = pt_trans(btpts, wparam_ori_to_halfhead_inv)

    return Calva_expand_pts_inori,Calva_stable_hairbt_pts_inori


    # return  ctlpts_inv




def getLinearEquation(p1,p2):
    x1, y1=tuple(p1)
    x2, y2=tuple(p2)

    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    return k,b


def cross_point(line1, line2):  # 计算交点函数
    x1,y1 = tuple(line1[0])  # 取四点坐标
    x2,y2 = tuple(line1[1])
    x3,y3 = tuple(line2[0])  # 取四点坐标
    x4,y4 = tuple(line2[1])

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3)==0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2==None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]



# def dot_product_angle(v1, v2):
#     if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
#         print("Zero magnitude vector!")
#         # return None
#     else:
#         vector_dot_product = np.dot(v1, v2)
#         arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
#         angle = np.degrees(arccos)
#         return angle
#     return 0

def get_cross_angle(l1, l2):
    arr_a = np.array(l1)  # 向量a
    arr_b = np.array(l2)  # 向量b
    cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
    eps = 1e-6
    if 1.0 < cos_value < 1.0 + eps:
        cos_value = 1.0
    elif -1.0 - eps < cos_value < -1.0:
        cos_value = -1.0
    return np.arccos(cos_value) * (180 / np.pi)  # 两个向量的夹角的角度， 余弦值：cos_value, np.cos(para), 其中para是弧度，不是角度


def get_cont2pts_by_cont1(cont_basept,cont1,cont2):
    cross_pts_list = []
    for etpt in cont1:
        # etpt=contpts_etou_instand[1]
        for k, hairoutpt in enumerate(cont2):
            if k==len(cont2) - 1:
                break
            cspt = cross_point([cont_basept, etpt], [cont2[k], cont2[k + 1]])

            vec1 = np.array(cspt) - np.array(cont2[k])
            vec2 = np.array(cspt) - np.array(cont2[k + 1])
            angle12 = get_cross_angle(vec1, vec2)
            if abs(angle12 - 180) < 0.001:
                cross_pts_list.append(cspt)
    return cross_pts_list

if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)
    matnet = init_matting_model()
    bise_net = init_parsing_model(model_name='bisenet')

    # srcroot = '/home/tao/mynas/Dataset/FaceEdit/sumiao/'
    # srcroot=r'/home/tao/mynas/Dataset/hairforsr/femalehd'
    # srcroot=r'/home/tao/Downloads/image_unsplash'
    srcroot=r'/home/tao/Downloads/unsplash_special'
    # srcroot=r'/home/tao/Pictures/test0'

    # srcroot='/home/tao/mynas/Dataset/FaceEdit/sumiao'
    # srcroot='/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/pexel_side_face'
    dstroot = '/home/tao/disk1/Workspace/TrainResult/eland/testim2/'

    ims = get_ims(srcroot)
    head_size = 2048

    etou_net=torch.jit.load('/home/tao/disk1/Workspace/TrainResult/eland/eland112-crop-resume3/plate_land_latest_jit.pt').to('cpu')

    trires=None


    ims.sort()
    for i, im in enumerate(ims):

        print(im)
        imname=os.path.basename(im)

        all_face_rcts = []
        all_face_lands = []
        frame = cv2.imread(im)
        image_const=np.array(frame)

        bbox_list, landmark_list = get_face_land_rct(det_net, align_net, image_const)

        img=cv2.imread(im)
        imgvis=np.array(img)
        landmark_list=[landmark_list[0]]

        for j,faceland_ori in  enumerate(landmark_list):

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

            Calva_base_pts = [faceland_ori[51]]
            etou_land_inori,backstable_quad_instand_inv=get_etou_ctlpts(img, faceland_ori,etou_net)
            Calva_expand_pts_inori,Calva_stable_hairbt_pts_inori=get_headout_ctlpts(img, faceland_ori, matnet, bise_net)
            # Calva_expand_pts_inori_result = expand_the_pts(Calva_base_pts, Calva_expand_pts_inori)
            Calva_expand_pts_inori_result=[]

            Calva_stable_hairbt_pts_inori_expanded = expand_the_pts_fixeratio(Calva_base_pts, Calva_stable_hairbt_pts_inori,1.5)


            ########################################################################################################
            contpts_etou=list(etou_land_inori)
            contpts_hairout=[Calva_stable_hairbt_pts_inori[0]]
            contpts_hairout.extend(Calva_expand_pts_inori)
            contpts_hairout.append(Calva_stable_hairbt_pts_inori[1])
            line_ptlist(imgvis, list(contpts_etou), (0, 0, 255), 10)
            line_ptlist(imgvis, list(contpts_hairout), (0, 0, 255), 10)

            contpts_etou_instand=pt_trans(contpts_etou,wparam_ori_to_standhead)
            contpts_hairout_instand = pt_trans(contpts_hairout, wparam_ori_to_standhead)
            faceland_instand=pt_trans(faceland_ori,wparam_ori_to_standhead)
            # cont_basept=faceland_ori[51]
            cont_basept=[0,0]
            cont_basept[1]=faceland_instand[54][1]
            cont_basept[0]=(contpts_etou_instand[0][0]+contpts_etou_instand[-1][0])/2

            # linek,lineb=getLinearEquation(cont_basept,contpts_etou_instand[3])

            cross_pts_list=[]
            for etpt in contpts_etou_instand:
                # etpt=contpts_etou_instand[1]
                for k,hairoutpt in enumerate(contpts_hairout_instand):
                    if k==len(contpts_hairout_instand)-1:
                        break
                    # cspt=cross_point([cont_basept,contpts_etou_instand[3]], [contpts_hairout_instand[k],contpts_hairout_instand[k+1]])
                    cspt = cross_point([cont_basept, etpt], [contpts_hairout_instand[k], contpts_hairout_instand[k + 1]])

                    vec1=np.array(cspt)-np.array(contpts_hairout_instand[k])
                    vec2 = np.array(cspt) - np.array(contpts_hairout_instand[k+1])
                    angle12=get_cross_angle(vec1,vec2)
                    if abs(angle12-180)<0.001:
                        cross_pts_list.append(cspt)


            contimg=np.zeros((2048,2048,3),dtype=np.uint8)
            draw_pts(contimg, list(merge_pts([[cont_basept],contpts_etou_instand])), 10, (0, 255, 0), 10)
            draw_pts(contimg, list(cross_pts_list), 10, (0, 255, 255), 10)

            cv2.imshow('contimg',limit_img_auto(contimg))



            # #####################################################################################################

            # etou_land_croped = pt_trans(etou_land, wparam_ori_to_etou)
            # Calva_back_stable_pts = make_backgroud_stable_pts(Calva_expand_pts_inori_result, Calva_stable_hairbt_pts_inori, Calva_base_pts)

            Calva_back_stable_pts=backstable_quad_instand_inv

            pt_src_list_inv = np.array(merge_pts([Calva_expand_pts_inori,Calva_base_pts,etou_land_inori,Calva_stable_hairbt_pts_inori_expanded,Calva_back_stable_pts]), np.int32)
            pt_dst_list_inv = np.array(merge_pts([Calva_expand_pts_inori_result , Calva_base_pts, etou_land_inori, Calva_stable_hairbt_pts_inori_expanded,Calva_back_stable_pts]), np.int32)


            # pt_src_list_inv = np.array(merge_pts([Calva_expand_pts_inori,Calva_base_pts,etou_land_inori,Calva_back_stable_pts]), np.int32)
            # pt_dst_list_inv = np.array(merge_pts([Calva_expand_pts_inori_result , Calva_base_pts, etou_land_inori,Calva_back_stable_pts]), np.int32)
            if trires is None:
                trires=get_trires(pt_src_list_inv)
            # warp_result=warp_the_img(img, pt_src_list_inv, pt_dst_list_inv,trires)
            vistri,offx, offy=vis_delaunay(pt_src_list_inv, imgvis,trires)
            draw_pts(vistri, list(np.array(etou_land_inori)+[offx, offy]), 20, (0, 0, 255), 10)
            draw_pts(vistri, list(np.array(Calva_expand_pts_inori) + [offx, offy]), 20, (255, 0, 0), 10)
            draw_pts(vistri, list(np.array(Calva_stable_hairbt_pts_inori) + [offx, offy]), 20, (255, 0, 0), 10)

            cv2.imshow('vistri',limit_img_auto(vistri))

            draw_pts(imgvis, list(faceland_ori), 10, (0, 255, 0), 10)
            draw_pts(imgvis, list(etou_land_inori), 20, (0, 0, 255), 10)
            draw_pts(imgvis, list(merge_pts([Calva_expand_pts_inori,Calva_expand_pts_inori_result ,Calva_stable_hairbt_pts_inori,Calva_back_stable_pts])), 20, (0, 255, 0), 10)


        cv2.imshow('img',limit_img_auto(imgvis))

        # show_img_two(img, warp_result)

        if keylist[0]==13:
            continue
        keylist[0] = cv2.waitKey(0)
        if keylist[0]==27:
            exit(0)