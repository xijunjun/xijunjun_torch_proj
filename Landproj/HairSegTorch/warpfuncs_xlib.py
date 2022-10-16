# coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform, math
# from delaunay2D import Delaunay2D
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

torch.set_grad_enabled(False)

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

def image_1to3c(imagein):
    image3c=np.array(imagein[:,:,None])
    image3c=image3c.repeat(3,axis=2)
    return  image3c

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
            exd_rct = np.array(bigger_rct(1.1, 1.1, rct), np.int32)
            faceimg, xshift, yshift = crop_pad(frame, exd_rct.reshape(2, 2))
            cv2.rectangle(frame, (exd_rct[0], exd_rct[1]), (exd_rct[2], exd_rct[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (rct[0], rct[1]), (rct[2], rct[3]), (0, 0, 255), 2)
            landmarks = align_net.get_landmarks(faceimg) + np.array([exd_rct[0], exd_rct[1]])
            landmarks = np.array(landmarks, np.int32)
            landmark_list.append(landmarks)
            bbox_list.append(rct)

    return bbox_list,landmark_list


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