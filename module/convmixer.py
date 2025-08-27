import torch
import torch.nn as nn
from einops import reduce

# Priority Channel Attention (PCA)
class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute the channel-wise mean of the input
        c = reduce(x, 'b c w h -> b c', 'mean')

        # Apply depthwise convolution
        x = self.dw(x)

        # Compute the channel-wise mean after convolution
        c_ = reduce(x, 'b c w h -> b c', 'mean')

        # Compute the attention scores
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(2).unsqueeze(3)  # Shape [batch_size, channels, 1, 1]
        return x * att_score  # Broadcasting to match the dimensions
    
# Priority Spatial Attention (PSA)
class PSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.prob = nn.Softmax2d()

    def forward(self, x):
        # Compute the spatial mean of the input
        s = reduce(x, 'b c w h -> b w h', 'mean')

        # Apply pointwise convolution
        x = self.pw(x)

        # Compute the spatial mean after convolution
        s_ = reduce(x, 'b c w h -> b w h', 'mean')

        # Compute the attention scores
        raise_sp = self.prob(s_ - s)
        att_score = torch.sigmoid(s_ * (1 + raise_sp))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(1)  # Shape [batch_size, 1, height, width]
        return x * att_score  # Broadcasting to match the dimensions

class activation_block(nn.Module):
  def __init__(self, outplane):
    super(activation_block, self).__init__()
    self.gelu = nn.GELU()
    self.outplane = outplane
    self.batchnorm = nn.BatchNorm2d(outplane)

  def forward(self, x):
    x = self.gelu(x)
    x = self.batchnorm(x)
    return x

class ConvMixer(nn.Module):
  def __init__(self, inplane,outplane):
    super(ConvMixer, self).__init__()
    self.inplane = inplane
    self.outplane = outplane
#     self.kernel_size = kernels_size
    self.depthwise = PCA(inplane)
    self.pointwise = PSA(outplane)
    self.activation = activation_block(outplane)
    self.filters = outplane
  def forward(self, x):
    #Depthwise convolution
    x0 = x
    x = self.depthwise(x)
    x = x + x0 #Residual
    #Pointwise convolution
    x = self.pointwise(x)
    x = self.activation(x)
    return x