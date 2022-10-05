
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



def img2tensor(img):
    img = img.transpose(2, 0, 1)
    imgtensor = torch.from_numpy(img)
    imgtensor=imgtensor.unsqueeze(0)
    return imgtensor


if __name__ == '__main__':
    # root='/home/tao/disk1/Workspace/TrainResult/eland/testim/'
    root='/home/tao/disk1/Workspace/TrainResult/eland/testim2/'
    checkptpath='/home/tao/disk1/Workspace/TrainResult/eland/eland112-crop-resume3/plate_land_latest.pt'

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


    ims=os.listdir(root)


    for im in ims:
        impath=root+im
        img=cv2.imread(impath).astype(np.float32)/255.0
        imgori=cv2.imread(impath)
        # cv2.imshow('img',img)
        img=cv2.resize(img,(insize,insize))

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        # out=net(img).cpu().numpy()
        land_pred=net(img)
        land_pred=land_pred.cpu().numpy()

        landmark = land_pred*1280

        print(land_pred.shape)

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

