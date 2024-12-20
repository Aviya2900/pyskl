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
        N, M, C, T, V= temp.size()
        temp = temp.view(N*M,C,T,V)#N*M,C,T,V
        x = temp.mean(-2)#N*M,C,V

        batch_size, C,  V = x.shape

        # x=x.permute(0,3,1,2).contiguous().view(N*V,C,T)#是一维的


        g_x = self.g(x).view(batch_size, self.in_channels, -1)

        g_x = g_x.permute(0, 2, 1) #N*M,V,C

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)#N*M,V,C
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)#N*M,C,V
        f = torch.matmul(theta_x, phi_x)#N*M,V,V

        dis = f.sum(-1)#N*M,V
        dis = dis * self.soft_scale

        dis = F.softmax(dis,dim=-1)#N*M,V

        argmax = torch.argmax(dis,dim=1)

        anchor = torch.einsum("bvc,bv->bc",g_x,dis) #N*M,C

        res = temp - anchor.unsqueeze(-1).unsqueeze(-1) #N*M,C,T,V
        res = res.view(N,M,C,T,V).permute(0)
        res = res.permute(2,0,3,1).contiguous().view(-1,C) #T,N*M,V,C -> #T*N*M*V,C

        # center_xyz = anchor.mean(-1).detach() # 注意这里把时间约去了
        center_xyz = anchor

        return res,dis,center_xyz

class SAP(nn.Module):
    def __init__(self,soft_scale=20,num_head = 5):
        super().__init__()
        self.left = nn.ModuleList([NonLocalBlockND(3,8,1,False,soft_scale=soft_scale) for i in range(num_head)])
        self.right = nn.ModuleList([NonLocalBlockND(3,8,1,False,soft_scale=soft_scale) for i in range(num_head)])
        self.head = num_head

    def angle_between(self,a, b):
        dot = torch.einsum("ni,ni->n", a, b)
        norma = torch.clamp(torch.norm(a, p=2, dim=1), min=1e-6)
        normb = torch.clamp(torch.norm(b, p=2, dim=1), min=1e-6)
        norm = norma * normb
        return dot / norm  # cos(angle)

    def forward(self,x):
        N, T, _ = x.shape
        anglelist, anchorsL, anchorsR = [], [], []
        for i in range(self.head):
            left, _, _ = self.left[i](x)
            right, _, _ = self.right[i](x)
            anchorsL.append(left)
            anchorsR.append(right)
            anglelist.append(self.angle_between(left,right).view(-1,1)) 
        
        anchorsL, anchorsR = torch.cat(anchorsL).view(N,2,T,-1,3,self.head), torch.cat(anchorsR).reshape(N,T,-1,3, self.head)
        anchors = torch.stack((anchorsL, anchorsR), dim=1)
        
        angle = torch.cat(anglelist, dim=1).view(N,T,2,-1,self.head)#.permute(1,4,0,3,2).contiguous()
        print(anchorsL.shape, anchorsR.shape, anchors.shape, angle.shape)
        
        #return angle#, anchors
    
if __name__ == "__main__":
    model = SAP()
    out = model(torch.randn(3,48,102))
    
    
