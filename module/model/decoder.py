import torch
import torch.nn as nn
from module.cbam import CBAM

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AxialMixer(nn.Module):
    def __init__(self, dim, mixer_kernel = (7,7), dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.mixer_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.mixer_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.mixer_h(x) + self.mixer_w(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, mixer_kernel = (7,7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.att = Attention_block(in_c, in_c, in_c//2)
        self.att2 = CBAM(in_c)
        self.pw = nn.Conv2d(in_c*2, in_c,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.dw = AxialMixer(in_c, mixer_kernel = mixer_kernel)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.pw(x)
        x = self.bn1(x)
        x = self.att2(x)
        x = self.act(self.bn2(self.pw2(self.dw(x))))
        return x