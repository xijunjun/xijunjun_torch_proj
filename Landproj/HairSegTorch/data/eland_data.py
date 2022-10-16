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

def load_hair_rct(txtpath):

    with open(txtpath,'r') as f:
        lines=f.readlines()
    line=lines[0]
    items=line.split(',')
    ptstr1,ptstr2=items[0],items[1]
    pt1=np.fromstring(ptstr1,dtype=np.int32,sep=' ')
    pt2 = np.fromstring(ptstr2, dtype=np.int32, sep=' ')
    return  pt1,pt2

def get_valid_impaths(imroot):

    ims=get_ims(imroot)
    validims=[]
    for i, im in enumerate(ims):
        imkey, ext=get_imkey_ext(im)
        txtpath=os.path.join(imroot,imkey+'.txt')
        if os.path.exists(txtpath) is False:
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
        pt1,pt2=load_hair_rct(txtpath)


        img = cv2.imread(impath)
        # imgsize=224
        halfhead=cv2.resize(img,(self.imgsize*2,self.imgsize))

        hairrct=np.array([pt1,pt2],np.int32)


        if random.random()<0.5:
            halfhead = cv2.flip(halfhead, 1)
            hairrct[0][0] = 1024 - hairrct[0][0]
            hairrct[1][0] = 1024 - hairrct[1][0]

        mask=np.zeros_like(img)
        cv2.rectangle(mask,tuple(hairrct[0]),tuple(hairrct[1]),(255,255,255),thickness=-1)
        mask=cv2.resize(mask, (self.imgsize*2, self.imgsize))
        mask=mask[:,:,0]


        # hairrct *=2
        # hairrct=hairrct.reshape(1,-1)[0]
        # hairrct/=2048

        return to_tensor(halfhead),to_tensor(mask)

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
    datadirlist = [
        # '/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/taobao_crop_good',
                   # '/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/hair_croped/helen_all_crop_good',
                   '/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/hair_croped/sumiao_crop_good'
                   ]

    wlfwdataset = ElandDataset(datadirlist=datadirlist,imgsize=128)
    traindata_loader = DataLoader(wlfwdataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    print(len(wlfwdataset))
    # exit(0)

    for img, mask in traindata_loader:
        print("img shape", img.shape)
        print('mask:',mask.shape)

        # img=img.asnumpy()
        img=to_pil_image(img[0])
        img=cv2.resize(img,(2048,1280))

        mask=to_pil_image(mask[0])
        mask=cv2.resize(mask,(2048,1280))


        cv2.imshow('img',img)
        cv2.imshow('mask', mask)

        if cv2.waitKey(0)==27:
            break





