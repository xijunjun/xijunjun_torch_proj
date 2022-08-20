# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
# import torch
# from torch import nn as nn
# from torch.nn import functional as F
# import torchscope as scope

# from basicsr.utils.registry import ARCH_REGISTRY



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import nn as nn
from torch.nn import functional as F

# from basicsr.utils.registry import ARCH_REGISTRY
from arch_util import default_init_weights, make_layer, pixel_unshuffle
from torchscope import scope

# @ARCH_REGISTRY.register(suffix='basicsr')
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=128, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out



def set_opt_param(optimizer, key, value):
    for group in optimizer.param_groups:
        group[key] = value



if __name__ == '__main__':

    device='cuda:0'

    net = SRVGGNetCompact(3,3,upscale=1).to('cuda')
    net.eval()
    x = torch.zeros([1, 3,512,512], dtype=torch.float).to('cuda')
    out=net(x)


    # #########################################################
    resume_checkpoint='//disks/disk1/Workspace/Project/Pytorch/TrainTemplate/basictest/tsbd/vgg.pth'
    pretrained_dict = torch.load(resume_checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    pre_keys=pretrained_dict.keys()
    net_keys=net.state_dict().keys()
    common_keys=list(set(pre_keys)&set(net_keys))
    new_check=net.state_dict()


    print(len(common_keys))
    for key in common_keys:
        if new_check[key].shape==pretrained_dict[key].shape:
            new_check[key]=pretrained_dict[key]
        else:
            print('ignore key:',key)

    net.load_state_dict(new_check)
    # ######################################################










    # torch.save(net.state_dict(), './tsbd/vgg.pth')

    # print(out.shape)
    # print(type(net))

    # with torch.no_grad():
    #     scope(net, input_size=(3, 1280, 1536),device='cuda')
