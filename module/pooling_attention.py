import torch.nn as nn

class Pooling_attention(nn.Module):
  def __init__(self,channel,kernel_size=5):
    super(Pooling_attention, self).__init__()
    self.adapavg = nn.AdaptiveAvgPool2d(1)
    self.adapmax = nn.AdaptiveMaxPool2d(1)
    self.conv = nn.Conv1d(1,1,kernel_size=kernel_size,padding=kernel_size//2)
    self.norm = nn.Sequential(
        # nn.BatchNorm2d(channel),
        nn.Sigmoid()
    )
  def forward(self,x):
    b,c,h,w = x.shape
    out1 = self.adapavg(x)
    out1 = self.conv(out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

    out2 = self.adapmax(x)
    out2 = self.conv(out2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


    out = out1+out2
    out = self.norm(out)
    # out = F.interpolate(out, size=(h//2, w//2), mode='bilinear', align_corners=False)
    out3 = nn.AdaptiveAvgPool2d((h//2,w//2))(x)
    out4 = nn.AdaptiveMaxPool2d((h//2,w//2))(x)
    out  = out*(out3+out4)
    return out