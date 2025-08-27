import torch.nn as nn

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class Attention_img(nn.Module):
  """Apply attention mechanism to output images"""
  def __init__(self):
    super().__init__()
    self.bn = nn.BatchNorm2d(1)
    self.relu  = nn.ReLU(inplace=True)
    self.sigmoid = nn.Sigmoid()
    self.conv_out = nn.Conv2d(2,1,kernel_size=1, bias=True)
    
  def forward(self, x1, x2):
    x = self.bn(x1 + x2)
    x = self.relu(x)
    x = self.sigmoid(x)
    return x1*x+x2