# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from .resnet3d import ResNet3d
from .SAP import SAP, NonLocalBlock

import torch
import torch.nn as nn

@BACKBONES.register_module()
class ResNet3dSlowOnly(ResNet3d):
    """SlowOnly backbone based on ResNet3d.

    Args:
        conv1_kernel (tuple[int]): Kernel size of the first conv layer. Default: (1, 7, 7).
        inflate (tuple[int]): Inflate Dims of each block. Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Other keywords arguments for 'ResNet3d'.
    """

    def __init__(self, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1), **kwargs):
        super().__init__(conv1_kernel=conv1_kernel, inflate=inflate, **kwargs)
        
        
@BACKBONES.register_module()
class ResNet3dSlowOnly_SAP(ResNet3d):

    def __init__(self, **kwargs):
        super().__init__()
        
        self.conv3d = nn.Conv3d(in_channels=17, out_channels=17, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=(1, 28, 11.2), mode='trilinear', align_corners=True)
        
        self.resnet3d = ResNet3dSlowOnly(**kwargs)
        self.gen_angle = SAP(soft_scale=20, num_heads=5)
        self.non_local = NonLocalBlock(in_channels=5, inter_channels=8)
        
    def forward(self, imgs, keypoints):
        N,M,T,V,C = keypoints.shape
        keys = torch.cat((keypoints, torch.ones(N, M, T, V, 1).to(keypoints.device)), dim=4) # N,M,T,V,C
        keys, _ = self.gen_angle(keys) # N,M,T,V,C'
        w_hmaps = imgs*self.correlation(imgs, keys) + imgs
                
        return self.resnet3d(w_hmaps)
        
    def correlation(self, x, y):
        N,M,T,V,C = y.shape # N,M,T,V,C'
        
        y = y.permute(0,2,1,3,4).contiguous() # N,T,M,V,C'
        y = (self.non_local(y)[1] + y).permute(0,3,1,2,4).contiguous() # N,V,T,M,C'
        w = self.conv3d(y) # N,V,T,V,C'
        w = self.upsample(w) # N,V,T,H,W

        return w

if __name__ == "__main__":
    model = ResNet3dSlowOnly_SAP(in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2))
    
    imgs = torch.randn(2,17,48,56,56)
    keys = torch.randn(2,2,48,17,2)
    
    out = model(imgs, keys)
    
    print(out.shape)