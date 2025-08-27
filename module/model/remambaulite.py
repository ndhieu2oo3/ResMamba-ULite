import torch.nn as nn
import torch.nn.functional as F
from module.model.encoder import EncoderBlock
from module.model.decoder import DecoderBlock
from module.csam import CSAM
from module.convmixer import ConvMixer
from module.decoder_attention import MapReduce, Attention_img

class ResMambaULite(nn.Module):
    def __init__(self):

        super().__init__()
        self.pw_in = nn.Conv2d(3, 16, kernel_size=1)

        """Encoder"""
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        """Skip connection"""

        self.c1 = CSAM(32)
        self.c2 = CSAM(64)
        self.c3 = CSAM(128)
        self.c4 = CSAM(256)
        self.c5 = CSAM(512)

        """Bottle Neck"""
        self.bridge = nn.Sequential(
            ConvMixer(512,512),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )

        """Decoder"""
        self.d5 = DecoderBlock(512, 512, 256)
        self.d4 = DecoderBlock(256, 256, 128)
        self.d3 = DecoderBlock(128, 128, 64)
        self.d2 = DecoderBlock(64, 64, 32)
        self.d1 = DecoderBlock(32, 32, 16)

        """Map Reduce"""
        self.map1 = MapReduce(256)
        self.map2 = MapReduce(128)
        self.map3 = MapReduce(64)
        self.map4 = MapReduce(32)
        self.map5 = MapReduce(16)
        """Decoder Attention"""
        self.att = Attention_img()

    def forward(self, x):

        H, W = x.shape[2], x.shape[3]

        """Encoder"""
        x = self.pw_in(x)

        x, skip1 = self.e1(x)

        x, skip2 = self.e2(x)

        x, skip3 = self.e3(x)

        x, skip4 = self.e4(x)

        x, skip5 = self.e5(x)

        """Skip Connection"""
        skip1 = self.c1(skip1)
        skip2 = self.c2(skip2)
        skip3 = self.c3(skip3)
        skip4 = self.c4(skip4)
        skip5 = self.c5(skip5)

        """BottleNeck"""

        x=self.bridge(x)

        """Decoder"""
        x = self.d5(x, skip5)
        x1 = self.map1(x)
        x1 = F.interpolate(x1, (H, W), mode="bilinear", align_corners=False)

        x = self.d4(x, skip4)
        x2 = self.map2(x)
        x2 = F.interpolate(x2, (H, W), mode="bilinear", align_corners=False)

        x = self.d3(x, skip3)
        x3 = self.map3(x)
        x3 = F.interpolate(x3, (H, W), mode="bilinear", align_corners=False)

        x = self.d2(x, skip2)
        x4 = self.map4(x)
        x4 = F.interpolate(x4, (H, W), mode="bilinear", align_corners=False)

        x = self.d1(x, skip1)
        x5 = self.map5(x)
        x5 = F.interpolate(x5, (H, W), mode="bilinear", align_corners=False)

        x4 = self.att(x4, x5)
        x3 = self.att(x3, x4)
        x2 = self.att(x2, x3)
        x1 = self.att(x1, x2)
        return x1