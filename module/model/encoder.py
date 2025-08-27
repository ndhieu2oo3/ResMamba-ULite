import torch.nn as nn
from module.resmambalite import ResMambaLite
from module.pooling_attention import Pooling_attention

class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c):
        super().__init__()

        self.pw= nn.Conv2d(in_c, out_c, kernel_size=3, padding = 'same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.mamba = ResMambaLite(in_c = in_c)
        self.pooling_att = Pooling_attention(out_c)

    def forward(self, x):
        x = self.mamba(x)
        skip = self.act(self.bn(self.pw(x)))
        x = self.pooling_att(skip)

        return x, skip