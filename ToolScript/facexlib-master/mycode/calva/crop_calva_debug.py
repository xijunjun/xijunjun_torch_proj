# coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform, math

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
    h, w, c = img.shape
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
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
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
    lrgap=40
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

    ims = get_ims(srcroot)
    # face_size = 2048
    face_size = 2536





    for i, im in enumerate(ims):
        all_face_rcts = []
        all_face_lands = []
        frame = cv2.imread(im)
        image_const=np.array(frame)

        bbox_list, landmark_list = get_face_land_rct(det_net, align_net, image_const)

        for j,landmarks in enumerate(landmark_list):
            land5_from98=land98to5(landmarks)

            warp_param_face_2048,warp_param_face_inv=get_crop_param(land5_from98)
            facealign = cv2.warpAffine(image_const, warp_param_face_2048, (face_size, face_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
            h,w,c=facealign.shape

            face_mat = get_mat(matnet, facealign)
            seg_bise = pred_seg_bise(bise_net, facealign)


            land98_in_crop=pt_trans(landmarks,warp_param_face_2048)
            for pt in land98_in_crop:
                pt=np.array(pt,np.int32)
                cv2.circle(facealign, (pt[0], pt[1]), 10, (255, 0, 0), -1, -1)

            calva_bottom_y=int(get_calva_bottom(land98_in_crop))

            print(warp_param_face_2048)
            face_mat_3c = image_1to3c(face_mat)

            calva_limit_rct,calva_crop_rct=get_calva_crop_rct(facealign, face_mat_3c, land98_in_crop)

            # cv2.rectangle(facealign, (calva_limit_rct[0], calva_limit_rct[1]), (calva_limit_rct[2], calva_limit_rct[3]), (0, 255, 255), 10)
            # cv2.rectangle(facealign, (calva_crop_rct[0], calva_crop_rct[1]), (calva_crop_rct[2], calva_crop_rct[3]), (0, 0, 255), 10)

            calva_crop_quad=[[calva_crop_rct[0],calva_crop_rct[1]],[calva_crop_rct[2],calva_crop_rct[1]],[calva_crop_rct[2],calva_crop_rct[3]],[calva_crop_rct[0],calva_crop_rct[3]]]
            # calva_crop_rct_origin=[]
            calva_quad_inv=pt_trans(list(calva_crop_quad) ,warp_param_face_inv)
            # for k in range(0, 4):
            #     calva_quad_inv.append()
            calva_quad_inv=np.array(calva_quad_inv,np.int32)

            for k in range(0,4):
                # cv2.line(facealign, tuple(calva_crop_quad[k % 4]), tuple(calva_crop_quad[(k + 1) % 4]), (0, 255, 0), 10)
                cv2.line(frame, tuple(calva_quad_inv[k % 4]), tuple(calva_quad_inv[(k + 1) % 4]), (0, 255, 0), 10)


            calva_dst_quad=np.array([[0,0],[1536,0],[1536,1280],[0,1280]])
            cava_crop_param = cv2.estimateAffinePartial2D(calva_quad_inv, calva_dst_quad, method=cv2.LMEDS)[0]
            calva_croped = cv2.warpAffine(image_const, cava_crop_param, (1536, 1280), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))


            # cv2.imshow('seg_bise', limit_img_auto(np.concatenate([facealign,face_mat_3c, matbin,matbin_sim], axis=1)))
            # cropimg, xshift, yshift=crop_pad(facealign,croprct.reshape(2,2))
            # h,w,c=cropimg.shape
            # print(cropimg.shape,1536/1280,w/h)
            # cv2.imshow('cropimg',limit_img_auto(cropimg))


        # landmarks = align_net.get_landmarks(frame)
        # landmarks = landmarks.astype(np.int32)

        # cv2.imwrite(dstroot+os.path.basename(im),facealign)

        cv2.imshow("capture", limit_img_auto(frame))
        cv2.imshow("facealign", limit_img_auto(facealign))
        cv2.imshow('calva_croped',limit_img_auto(calva_croped))

        # if cv2.waitKey(10) & 0xff == ord('q'):
        #     break
        key = cv2.waitKey(0)
        if key==27:
            break
