
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


from network import *
# from  mbv3_net import *

from warpfuncs_xlib import *



def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



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

def pt_trans(pts,param):

    dst=[]
    for pt in pts:
        x = pt[0]*param[0][0]+pt[1]*param[0][1]+param[0][2]
        y = pt[0] * param[1][0] + pt[1] * param[1][1] + param[1][2]
        dst.append([x,y])
    return  np.array(dst)

def get_crop_param_targetsize(landpts5,targetsize):
    template_2048=np.array([[863,1147],[1217,1147],[1043,1383],[889,1547],[1193,1547]])
    template_2048 +=244
    template_nrom=template_2048/2536
    head_temp=template_nrom*targetsize

    warp_param_face_2048=cv2.estimateAffinePartial2D(landpts5, head_temp, method=cv2.LMEDS)[0]
    warp_param_face_inv=cv2.invertAffineTransform(warp_param_face_2048)

    return warp_param_face_2048,warp_param_face_inv

def img2bin_uint(imgin):
    img=np.array(imgin)
    thres=100
    img[img<thres]=0
    img[img>=thres]=255
    return   img

def simplify_mask(maskin):

    contours, h = cv2.findContours(maskin.copy()[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1:
        return  None

    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    # img = cv2.drawContours(maskin.copy(), [contours[0]], -1, (0, 255, 0), 5)
    img = cv2.drawContours(np.zeros_like(maskin), [contours[0]], -1, (255, 255, 255), -1)
    return img


def get_imkey_ext(imname):
    imname=os.path.basename(imname)
    ext='.'+imname.split('.')[-1]
    imkey=imname.replace(ext,'')
    return imkey,ext



def rename_all(srcroot):
    ims = get_ims(srcroot)
    for i,im in enumerate(ims):

        imname=os.path.basename(im)
        imkey,ext=get_imkey_ext(imname)

        os.rename(im,os.path.join(srcroot,'taobao_download_'+str(i).zfill(5)+ext))
        print(im,os.path.join(srcroot,'taobao_download_'+str(i).zfill(5)+ext))



def makedir(dirtp):
    if os.path.exists(dirtp):
        return
    os.makedirs(dirtp)



def img2tensor(img):
    img = img.transpose(2, 0, 1)
    imgtensor = torch.from_numpy(img)
    imgtensor=imgtensor.unsqueeze(0)
    return imgtensor


def img2tensor(img):
    img = img.transpose(2, 0, 1)
    imgtensor = torch.from_numpy(img)
    imgtensor=imgtensor.unsqueeze(0)
    return imgtensor

def pred_etou_land(etou_net,imgin):
    device='cpu'

    img=np.array(imgin,np.float32)
    img=cv2.resize(img,(112,112))

    land_pred = etou_net(img2tensor(img).to(device))
    land_pred = land_pred.cpu().numpy()
    etouland = land_pred * 2048
    return np.array(etouland[0],np.int32)

if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)



    checkptpath='/home/tao/disk1/Workspace/TrainResult/hairrct/hairrct112-resume3/plate_land_latest.pt'
    insize=112
    torch.set_grad_enabled(False)
    device = 'cuda:1'
    net = PFLDInference()
    # Step 2: model, criterion, optimizer, scheduler
    net = net.to(device)

    net = load_model(net,checkptpath,False)
    net.eval()
    net_jit = torch.jit.trace(net, img2tensor(np.zeros((insize,insize,3),dtype=np.float32)).to(device))
    net_jit.save(checkptpath.replace('.pt','_jit.pt'))

    checkptpath=checkptpath.replace('.pt','_jit.pt')

    # srcroot='/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/taobao'
    # srcroot=r'/home/tao/mynas/Dataset/FaceEdit/sumiao'
    srcroot=r'/home/tao/Downloads/portrait'

    torch.set_grad_enabled(False)
    device = 'cpu'
    etou_net=torch.jit.load(checkptpath)

    etou_net = etou_net.to(device)



    dstroot=srcroot+'_crop'

    makedir(dstroot)

    # rename_all(srcroot)

    # exit(0)


    ims = get_ims(srcroot)
    # face_size = 2048
    # head_size = 2536
    head_size = 2048

    for i, im in enumerate(ims):
        imname=os.path.basename(im)
        imkey, ext = get_imkey_ext(imname)

        all_face_rcts = []
        all_face_lands = []
        frame = cv2.imread(im)
        image_const=np.array(frame)

        bbox_list, landmark_list = get_face_land_rct(det_net, align_net, image_const)

        img=cv2.imread(im)
        imgvis=np.array(img)



        for j,landmarks in enumerate(landmark_list):
            path_imdst = os.path.join(dstroot, imkey + '_' + str(j) + '.jpg')
            if os.path.exists(path_imdst) is True:
                break

            frame_vis = np.array(frame)
            ##########裁剪出单张人头
            land5_from98=land98to5(landmarks)
            # warp_param_head,warp_param_face_inv=get_crop_param(land5_from98)
            warp_param_head,warp_param_face_inv=get_crop_param_targetsize(land5_from98, head_size)

            headalign = cv2.warpAffine(image_const, warp_param_head, (head_size, head_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
            headalign_vis=np.array(headalign)
            h,w,c=headalign.shape

            print(im,' headalign.shape：',headalign.shape)

            # print('facealign.shape ',facealign.shape)


            ##########获取单张人头的关键点
            land98_in_crop=pt_trans(landmarks,warp_param_head)
            # for pt in land98_in_crop:
            #     pt=np.array(pt,np.int32)
            #     cv2.circle(headalign, (pt[0], pt[1]), 10, (255, 0, 0), -1, -1)

            ##########获取颅顶裁剪框

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


            param_halfhead_inori = cv2.estimateAffinePartial2D(halfhead_quad_inv, halfhead_dst_quad, method=cv2.LMEDS)[0]
            param_halfhead_inori_inv=cv2.invertAffineTransform(param_halfhead_inori)
            landmark_in_half=pt_trans(landmarks,param_halfhead_inori)



            # print(land_pred.shape)

            # print(landmark)

            halfheadalign=halfheadalign.astype(np.float32)/255.0

            landmark = pred_etou_land(etou_net, halfheadalign)

            print(landmark)

            for i in range(0, 2):
                cv2.circle(halfheadalign, (int(landmark[i * 2]), int(landmark[i * 2 + 1])), 30, (0, 1, 1), 10)

            cv2.rectangle(halfheadalign, (landmark[0],landmark[1]), (landmark[2],landmark[3]), (0, 0, 1), 10, 1)

            cv2.imshow('halfheadalign ',limit_img_auto(halfheadalign ))
            if cv2.waitKey(0)==27:
                exit(0)
















