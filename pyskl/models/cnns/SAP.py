import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, soft_scale=20):
        super(NonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1), 
                               nn.BatchNorm2d(self.inter_channels), nn.ReLU())
        self.theta = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1), 
                                   nn.BatchNorm2d(self.inter_channels), nn.ReLU())
        self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels ,out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1), 
                                 nn.BatchNorm2d(self.inter_channels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1), 
                                 nn.BatchNorm2d(self.in_channels))
        
        self.softmax = F.softmax
        
    def forward(self, x):
        N,M,T,V,C = x.shape
        x = x.permute(0,1,4,2,3).contiguous().view(-1,C,T,V) # N*M,C,T,V
        
        theta_x = self.theta(x).view(N*M,self.inter_channels,T*V) # N*M,8(self.inter_channels),T,V -> # N*M,8(self.inter_channels),T*V
        phi_x = self.phi(x).view(N*M,self.inter_channels,T*V) # N*M,8(self.inter_channels),T,V -> # N*M,8(self.inter_channels),T*V
        g_x = self.g(x).view(N*M,self.inter_channels,T*V) # N*M,8(self.inter_channels),T,V -> # N*M,8(self.inter_channels),T*V
        
        att = torch.bmm(theta_x.permute(0,2,1), phi_x) # N*M,T*V,T*V
        att = self.softmax(att, dim=-1) # N*M,T*V,T*V
        
        anchors = torch.bmm(g_x, att.permute(0,2,1).contiguous()) # N*M,8(self.inter_channels),T*V
        anchors = self.out(anchors.view(N*M,self.inter_channels,T,V)) # N*M,3(self.in_channels),T,V
        
        anchors = anchors.permute(0,2,3,1).contiguous().view(N,M,T,V,C) # N,M,T,V,C
        x = x.permute(0,2,3,1).contiguous().view(N,M,T,V,C) # N,M,T,V,C
        dirs = anchors-x # N,M,T,V,C
        
        return dirs, anchors
    
class SAP(nn.Module):
    def __init__(self, soft_scale, num_heads, in_channels=3, inter_channels=8):
        super(SAP, self).__init__()
        
        self.num_heads = num_heads
        self.soft_scale = soft_scale
        self.left = nn.ModuleList([NonLocalBlock(in_channels=in_channels, inter_channels=inter_channels, soft_scale=soft_scale) for i in range(num_heads)])
        self.right = nn.ModuleList([NonLocalBlock(in_channels=in_channels, inter_channels=inter_channels, soft_scale=soft_scale) for i in range(num_heads)])
    
    def forward(self, x):
        N,M,T,V,C = x.shape
        
        angles, left_xyz, right_xyz = [], [], []
        for i in range(self.num_heads):
            left, right = self.left[i](x), self.right[i](x) # N,M,T,V,C(inter_channels)            
            angles_ = self.angle_between(left[0], right[0]) # N,M,T,A (A=angle in rad)
            angles.append(angles_)
            left_xyz.append(left[1])
            right_xyz.append(right[1])     
        anchors = torch.concat([torch.stack(left_xyz, dim=4), torch.stack(right_xyz, dim=4)], dim=4) # N,M,T,V,2*num_heads,C
        angles = torch.stack(angles, dim=4) # N,M,T,V,num_heads
        
        return angles, anchors
            
    def angle_between(self, left, right):
        cross = torch.cross(left, right, dim=-1)
        dot = torch.sum(left * right, dim=-1)
        angles = torch.atan2(torch.norm(cross, dim=-1), dot)
        
        return angles
    
if __name__ == "__main__":
    lin = torch.randn(5,2,48,17,3)
    model = NonLocalBlock(3,8,20.0)
    out = model(lin)
    #print(out[0].shape, out[1].shape)
    model = SAP(soft_scale=20, num_heads=5)
    out = model(lin)
    print(out[0].shape, out[1].shape)
    
        