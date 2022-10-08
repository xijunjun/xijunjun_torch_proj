import os.path
from os.path import join
from torch.utils import data
from PIL import Image
import random
import torch
import math
import  numpy   as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import  cv2

import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()

def to_pil_image(input):
    return np.array(transforms.ToPILImage()(input))

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

def load_calva_land(txtpath):
    with open(txtpath,'r') as f:
        lines=f.readlines()
    faceland=str2pts(lines[0])
    portait_ext_land = str2pts(lines[1])
    etou_land = str2pts(lines[2])
    return  faceland,portait_ext_land,etou_land

def load_etou_land(txtpath):
    with open(txtpath,'r') as f:
        lines=f.readlines()
    faceland = str2pts(lines[0])
    etou_land = str2pts(lines[2])
    return  faceland,etou_land

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

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()
# ###########################################
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

def get_etou_crop_rct_byland(landmarks,headw):
    h, w, c = headw,headw,3
    landmarksnp=np.array(landmarks,np.int32)
    stand_etouw=1280
    stand_etouh=960
    stand_etouwh_ratio=1.0*stand_etouw/stand_etouh
    # wextend_ratio=1.8
    # hextend_ratio = 1.8

    if random.random()<0.6:
        wextend_ratio=1.8
        hextend_ratio = 1.8
    else:
        wextend_ratio=1.6+random.random()*0.3
        hextend_ratio = 1.4+random.random()*0.6


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

    # print('extend_w /extend_h',extend_w ,extend_h)
    # print('stand_etouwh_ratio,cur_wh_ratio:',stand_etouwh_ratio,cur_wh_ratio)

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


def get_crop_param_targetsize(landpts5,targetsize):
    template_2048=np.array([[863,1147],[1217,1147],[1043,1383],[889,1547],[1193,1547]])
    template_2048 +=244
    template_nrom=template_2048/2536
    head_temp=template_nrom*targetsize

    warp_param_face_2048=cv2.estimateAffinePartial2D(landpts5, head_temp, method=cv2.LMEDS)[0]
    warp_param_face_inv=cv2.invertAffineTransform(warp_param_face_2048)

    return warp_param_face_2048,warp_param_face_inv




class ElandDataset(data.Dataset):
    def __init__(self, datadirlist, imgsize=112):
        super(ElandDataset, self).__init__()

        self.validims=[]
        for datadir in datadirlist:
            self.validims.extend(get_valid_impaths(datadir))
        self.imgsize=imgsize


        # print(self.fns)

    def __getitem__(self, index):
        impath, txtpath = self.validims[index]
        faceland,etou_land=load_etou_land(txtpath)

        img = cv2.imread(impath)


        land5_from98 = land98to5(faceland)
        # 完整人头裁剪参数
        warp_param_head, warp_param_face_inv = get_crop_param_targetsize(land5_from98, 2048)
        faceland_in_crop=pt_trans(faceland,warp_param_head)
        etou_land_in_crop=pt_trans(etou_land,warp_param_head)
        # 获取额头裁剪框
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
        etou_land_croped=np.array(etou_land_croped,np.float32)
        etou_croped= cv2.warpAffine(img, param_etoucroped_inori, (etouw, etouh), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))

        # imgsize=224
        etou_croped=cv2.resize(etou_croped,(self.imgsize,self.imgsize))

        # etou_land_croped=etou_land_croped.T
        etou_land_croped=etou_land_croped.reshape(1,-1)[0]

        etou_land_croped/=1280

        return to_tensor(etou_croped),etou_land_croped

    def __len__(self):
        return len(self.validims)


class PFLDDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None, img_root=None, img_size=112):
        assert img_root is not None
        self.line = None
        self.path = None
        self.img_size = img_size
        self.landmarks = None
        self.filenames = None
        self.euler_angle = None
        self.img_root = img_root
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(os.path.join(self.img_root, self.line[0]))
        self.img = cv2.resize(self.img, (self.img_size, self.img_size))
        self.landmark = np.asarray(self.line[1:213], dtype=np.float32)
        self.euler_angle = np.asarray(self.line[213:], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return self.img, self.landmark, self.euler_angle

    def __len__(self):
        return len(self.lines)




def draw_pts(img,ptlist,r,color,thick,wait=0):
    for pt in ptlist:
        cv2.circle(img,tuple(np.array(pt,np.int32)),r,color,thick)

if __name__ == '__main__':

    # wlfwdataset = ElandDataset(datadir='/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/ffhq')
    # dataloader = DataLoader(wlfwdataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    #
    #
    datadirlist=[
        # '/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/ffhq',
                 '/home/tao/disk1/Dataset/Project/FaceEdit/etou_data/ffhq2k'
    ]

    wlfwdataset = ElandDataset(datadirlist=datadirlist,imgsize=112)
    traindata_loader = DataLoader(wlfwdataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    print(len(wlfwdataset))
    exit(0)

    for img, landmark in traindata_loader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())
        # img=img.asnumpy()
        img=to_pil_image(img[0])
        img=cv2.resize(img,(1280,960))
        landmark*=1280

        # print(type(img),img.shape)
        # exit(0)

        # draw_pts(img, list(landmark[0]), 10, (0, 255, 0), 10)
        for i in range(0,20):
            cv2.circle(img, (int(landmark[0][i*2]),int(landmark[0][i*2+1])), 10, (0, 255, 0), 10)

        cv2.imshow('img',img)

        if cv2.waitKey(0)==27:
            break





