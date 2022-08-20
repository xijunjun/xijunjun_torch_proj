import cv2
import torch
import util.util as util
import models
import time
import os
import sys
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np

to_tensor = transforms.ToTensor()
def to_pil_image(input):
    return np.array(transforms.ToPILImage()(input))



def cvt_tensor_color(imtensor_):
    imtensor=torch.tensor(imtensor_)
    temp=torch.tensor(imtensor[0])
    imtensor[0]=imtensor[2]
    imtensor[2]=temp

    print(temp.shape)
    return  imtensor

class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.__setup()

    def __setup(self):
        opt = self.opt
        
        """Model"""
        self.model = models.__dict__[self.opt.model]()
        self.model.initialize(opt)
        # if not opt.no_log:
        #     self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))

        self.writer = SummaryWriter(log_dir=self.opt.exp_tbs_root)


    def train(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            iterations = self.iterations
            

            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)
            
            errors = model.get_current_errors()
            avg_meters.update(errors)
            # util.progress_bar(i, len(train_loader), str(avg_meters))
            print(i, len(train_loader), str(avg_meters))
            
            # if not opt.no_log:
            #     util.write_loss(self.writer, 'train', avg_meters, iterations)
            
                # if iterations % opt.display_freq == 0 and opt.display_id != 0:
                #     save_result = iterations % opt.update_html_freq == 0

                # if iterations % opt.print_freq == 0 and opt.display_id != 0:
                #     t = (time.time() - iter_start_time)

            visualim_dict=model.get_current_visuals()

            for imkey in visualim_dict.keys():
                self.writer.add_image(imkey, to_tensor(cv2.cvtColor(visualim_dict[imkey],cv2.COLOR_BGR2RGB)), i)


            self.iterations += 1
    
        self.epoch += 1


        if self.epoch % opt.save_epoch_freq == 0:
            print('saving the model at epoch %d, iters %d' %
                (self.epoch, self.iterations))
            model.save()

        print('saving the latest model at the end of epoch %d, iters %d' %
            (self.epoch, self.iterations))
        model.save(label='latest')

        print('Time Taken: %d sec' %
            (time.time() - epoch_start_time))
                
        # model.update_learning_rate()
        train_loader.reset()


    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
