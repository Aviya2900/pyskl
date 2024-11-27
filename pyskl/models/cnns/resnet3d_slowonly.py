# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from .resnet3d import ResNet3d

import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True,\
                 num_frame=300,soft_scale=20):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.soft_scale=soft_scale


        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, input, return_nl_map=True):
        """
        :param x: N,T,-1
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return: T*N*M*V,3 #固定坐标轴
        """
        N,T,_ = input.shape
        temp = input.view(N, T, 2, -1, 3).permute(0, 2, 4, 1, 3).contiguous()
        N, M,C, T, V= temp.size()
        temp = temp.view(N*M,C,T,V)#N*M,C,T,V
        x = temp.mean(-2)#N*M,C,V

        batch_size, C,  V = x.shape

        # x=x.permute(0,3,1,2).contiguous().view(N*V,C,T)#是一维的


        g_x = self.g(x).view(batch_size, self.in_channels, -1)#

        g_x = g_x.permute(0, 2, 1)#N*M,V,C

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)#N*M,V,C
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)#N*M,C,V
        f = torch.matmul(theta_x, phi_x)#N*M,V,V

        dis = f.sum(-1)#N*M,V
        dis = dis * self.soft_scale

        dis = F.softmax(dis,dim=-1)#N*M,V

        argmax = torch.argmax(dis,dim=1)

        anchor = torch.einsum("bvc,bv->bc",g_x,dis)

        res = temp - anchor.unsqueeze(-1).unsqueeze(-1)#bctv
        res = res.permute(2,0,3,1).contiguous().view(-1,C)

        # center_xyz = anchor.mean(-1).detach() # 注意这里把时间约去了
        center_xyz = anchor.detach()

        return res,dis,center_xyz

class SAP(nn.Module):
    def __init__(self,soft_scale=20,num_head = 5):
        super().__init__()
        self.left = nn.ModuleList([NonLocalBlockND(3,8,1,False,soft_scale=soft_scale) for i in range(num_head)])
        self.right = nn.ModuleList([NonLocalBlockND(3,8,1,False,soft_scale=soft_scale) for i in range(num_head)])
        self.head = num_head

    def angle_betwenn(self,a, b):
        dot = torch.einsum("ni,ni->n", a, b)
        norma = torch.clamp(torch.norm(a, p=2, dim=1), min=1e-6)
        normb = torch.clamp(torch.norm(b, p=2, dim=1), min=1e-6)
        norm = norma * normb
        return dot / norm  # cos(angle)

    def forward(self,x):
        N, T, _ = x.shape
        anglelist = []
        leftargmax = []
        rightargmax = []
        leftxyz = []
        rightxyz = []
        for i in range(self.head):
            left, temp, tempxyz = self.left[i](x)
            leftargmax.append(temp)
            leftxyz.append(tempxyz)
            right, temp, tempxyz = self.right[i](x)
            rightargmax.append(temp)
            rightxyz.append(tempxyz)
            anglelist.append(self.angle_betwenn(left,right).view(-1,1))

        angle = torch.cat(anglelist,dim=1).view(T,N,2,-1,self.head).permute(1,4,0,3,2).contiguous()

        return angle\


@BACKBONES.register_module()
class ResNet3dSlowOnly_SAP(ResNet3d):

    def __init__(self, **kwargs):
        super().__init__()
            
        self.conv3d_1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(8),
                                    nn.ReLU())
        self.conv3d_2 = nn.Sequential(nn.Conv3d(in_channels=8, out_channels=17, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(17),
                                            nn.ReLU())
        self.upsample = nn.Upsample(size=(48, 56, 56), mode='trilinear', align_corners=True)
        
        self.resnet3d = ResNet3dSlowOnly(**kwargs)
        self.gen_angle = SAP()
        
    def forward(self, imgs, keypoints):
        N, M, T, V, _ = keypoints.shape
        keys = torch.cat((keypoints, torch.ones(N, M, T, V, 1).to(keypoints.device)), dim=4)
        keys = keys.permute(0,2,1,3,4).contiguous().view(N,T,-1) # N,T,M*V*C
        keys = self.gen_angle(keys)
        w_hmaps = self.correlation(imgs, keys) + imgs
                
        return self.resnet3d(w_hmaps)
        
    def correlation(self, x, y):
        N, V, T, _, _ = x.shape # N,V,T,H,W
        _, C, _, _, M = y.shape
        y = y.permute(0,2,4,3,1).contiguous().view(N,T,M*V,C) # N,T,M*V,C

        w = torch.matmul(y, y.permute(0,1,3,2).contiguous()).unsqueeze(1) # N,1,T,(M*V),(M*V)
        w = self.conv3d_1(w) # N,8,T,(M*V),(M*V)
        w = self.conv3d_2(w) # N,17,T,(M*V),(M*V)
        w = self.upsample(w) # N,V,T,H,W

        w_hmaps = (x*w) # N,V,T,H,W

        return w_hmaps
    
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
