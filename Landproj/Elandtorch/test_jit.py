
from __future__ import print_function
import os,cv2
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
import time
import datetime
import math
import torchvision.models._utils as _utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network import *
from data.eland_data import  ElandDataset





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
    etouland = land_pred * 1280
    return np.array(etouland,np.float32)


if __name__ == '__main__':
    # root='/home/tao/disk1/Workspace/TrainResult/eland/testim/'
    root='/home/tao/disk1/Workspace/TrainResult/eland/testim2/'
    checkptpath='/home/tao/disk1/Workspace/TrainResult/eland/eland00/plate_land_latest.pt'


    torch.set_grad_enabled(False)
    device = 'cpu'

    etou_net=torch.jit.load('/home/tao/disk1/Workspace/TrainResult/eland/eland00/plate_land_latest_jit.pt')

    etou_net = etou_net.to(device)

    ims=os.listdir(root)


    for im in ims:
        impath=root+im
        img=cv2.imread(impath).astype(np.float32)/255.0
        imgori=cv2.imread(impath)

        # img=cv2.resize(img,(112,112))
        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(device)
        # land_pred=etou_net(img)
        # land_pred=land_pred.cpu().numpy()
        # landmark = land_pred*1280

        landmark=pred_etou_land(etou_net, img)

        # print(land_pred.shape)

        print(landmark)

        for i in range(0,20):
            cv2.circle(imgori, (int(landmark[0][i*2]),int(landmark[0][i*2+1])), 10, (0, 255, 0), 10)

        cv2.imshow('img',imgori)

        if cv2.waitKey(0)==27:
            break


        # # print (label)
        # for j in range(4):
        #     cv2.line(imgori, (quad[j*2], quad[j*2+1]), (quad[(j+1)%4*2], quad[(j+1)%4*2+1]), (0, 0, 255), thickness=2)
        #
        # cv2.imshow('img', imgori)
        # # print(out.data)
        #
        # key=cv2.waitKey(0)
        # if key==27:
        #     exit(0)

