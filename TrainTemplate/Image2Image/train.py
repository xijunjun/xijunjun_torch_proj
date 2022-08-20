import shutil
from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
# import data.reflect_dataset as datasets
import data.inpainting_dataset as datasets

import util.util as util
import data
import sys
from config import cfg as opt
import argparse,os


# opt = TrainOptions().parse()

# cudnn.benchmark = True
#
# opt.display_freq = 10
#
# if opt.debug:
#     opt.display_id = 1
#     opt.display_freq = 20
#     opt.print_freq = 20
#     opt.nEpochs = 40
#     opt.max_dataset_size = 100
#     opt.no_log = False
#     opt.nThreads = 0
#     opt.decay_iter = 0
#     opt.serial_batches = True
#     opt.no_flip = True

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    """Main Loop"""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_base_root', type=str, default='./')
    parser.add_argument('--yaml_path', type=str, default='errnet_model')
    syscfg = parser.parse_args()

    # print(sys.argv[1])


    yamlpath= syscfg.yaml_path
    exp_base_root=syscfg.exp_base_root
    exp_checkpoints_root = os.path.join(exp_base_root, 'checkpoints')
    exp_logs_root = os.path.join(exp_base_root, 'logs')
    exp_tbs_root = os.path.join(exp_base_root, 'tensorboard')

    if os.path.exists(exp_tbs_root):
        shutil.rmtree(exp_tbs_root)
    mkdirs([exp_checkpoints_root, exp_logs_root, exp_tbs_root])

    opt.merge_from_file(yamlpath)
    opt.exp_checkpoints_root=exp_checkpoints_root
    opt.exp_logs_root=exp_logs_root
    opt.exp_tbs_root=exp_tbs_root



    engine = Engine(opt)


    def set_learning_rate(lr):
        for optimizer in engine.model.optimizers:
            print('[i] set learning rate to {}'.format(lr))
            util.set_opt_param(optimizer, 'lr', lr)

    traindataset=datasets.InpaintingDataset('/disks/disk1/Dataset/Project/SuperResolution/taobao_stand_face')

    train_dataloader = datasets.DataLoader(
        traindataset, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
        num_workers=opt.nThreads, pin_memory=True)

    # define training strategy
    # engine.model.opt.lambda_gan = 0
    # engine.model.opt.lambda_gan = 0.01
    set_learning_rate(1e-3)
    while engine.epoch < 60:
        if engine.epoch == 20:
            engine.model.opt.lambda_gan = 0.01  # gan loss is added after epoch 20
        if engine.epoch == 30:
            set_learning_rate(5e-5)
        if engine.epoch == 40:
            set_learning_rate(1e-5)
        if engine.epoch == 45:
            set_learning_rate(5e-5)
        if engine.epoch == 50:
            set_learning_rate(1e-5)

        engine.train(train_dataloader)

