from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.gpu_ids= [0,]
_C.exp_checkpoints_root=''
_C.exp_logs_root=''
_C.exp_tbs_root=''
_C.save_epoch_freq=1
_C.lambda_gan= 1.0
_C.lambda_vgg= 1.0
_C.lr=0.001
_C.wd=0
_C.inet='errnet'
_C.init_type='xavier'
_C.resume=False
_C.no_verbose= False
_C.icnn_path=''
_C.unaligned_loss='vgg'

_C.gan_type= 'rasgan'
_C.which_model_D= 'disc_patch'
_C.vgg_layer=31

_C.model='errnet_model'


_C.batchSize=1
_C.serial_batches=True
_C.nThreads=8
_C.isTrain=True
