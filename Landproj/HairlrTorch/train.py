


import argparse
import logging
from pathlib import Path
import os

import numpy as np
import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn

# from network import *
from  mbv3_net import *


from data.eland_data import  ElandDataset




class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmark_gt, landmarks):
        l2_distant = torch.sum((landmark_gt - landmarks) ** 2, axis=1)

        # loss_landm = F.smooth_l1_loss(land_pred, landm_t, reduction='sum')

        return torch.mean(l2_distant)

def makedir(dirtp):
    if os.path.exists(dirtp):
        return
    os.makedirs(dirtp)

if __name__ == '__main__':

    device='cuda:0'
    base_lr=0.001
    max_epoch=1000
    save_freq=20
    checkpointdir='/home/tao/disk1/Workspace/TrainResult/hairrct'
    project_name='mbv3_hairrct112'
    checkdir=checkpointdir+'/'+project_name
    makedir(checkdir)
    resume_checkpoint=None

    imgsize=112

    datadirlist=['/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/taobao_crop_good',
                 '/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/hair_croped/helen_all_crop_good',
                 '/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/hair_croped/sumiao_crop_good'
]

    # resume_checkpoint='/home/tao/disk1/Workspace/TrainResult/hairrct/hairrct112-resume2/plate_land_latest.pt'

    wlfwdataset = ElandDataset(datadirlist=datadirlist,imgsize=imgsize)
    traindata_loader = DataLoader(wlfwdataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    print('numims:',len(wlfwdataset))



    plfd_backbone = PFLDInference()
    # Step 2: model, criterion, optimizer, scheduler
    plfd_backbone = plfd_backbone.to(device)

    if resume_checkpoint is not None:
        pretrained_dict = torch.load(resume_checkpoint, map_location=lambda storage, loc: storage.cuda(device))

        filterd_dict=plfd_backbone.state_dict()
        keys=pretrained_dict.keys()
        # print(keys)

        # print(plfd_backbone.state_dict().keys())
        for key in keys:
            if pretrained_dict[key].shape==filterd_dict[key].shape:
                filterd_dict[key]=pretrained_dict[key]

        plfd_backbone.load_state_dict(filterd_dict)
        print('load resume_checkpoint:',resume_checkpoint)


    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([{'params': plfd_backbone.parameters()}],lr=base_lr,weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=40, verbose=True)



    # init_optimizer([optimizer_D])
    for epoch in range(0,max_epoch):
        i=0
        for data,land_gt in traindata_loader:
            # print (label)
            data = data.to(device=device)
            land_gt=land_gt.to(device=device)
            # print(label.shape)


            land_pred=plfd_backbone(data)
            # print(land_gt.shape,land_pred.shape)

            loss = criterion(land_gt, land_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # print(land_pred.shape)
            print(('epoch{} loss: {}').format(epoch,loss.item()))


        if epoch%save_freq==0 or epoch==max_epoch:
            savepath=checkpointdir+'/'+project_name+'/plate_land_'+str(epoch).zfill(4)+'.pt'
            torch.save(plfd_backbone.state_dict(), savepath)
            print('save model:',savepath)

        if epoch % save_freq==0:
            savepath = checkpointdir+'/' + project_name + '/plate_land_latest.pt'
            torch.save(plfd_backbone.state_dict(), savepath)
            print('save latest model')

