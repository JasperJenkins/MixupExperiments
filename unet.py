import torch
import torch.nn as nn

def conv(in_c, out_c):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(out_c),
    nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(out_c),
  )

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    self.down1 = conv(  2,  16) # (  2, 128, 128) --> ( 16, 128, 128)
    self.down2 = conv( 16,  32) # ( 16,  64,  64) --> ( 32,  64,  64)
    self.down3 = conv( 32,  64) # ( 32,  32,  32) --> ( 64,  32,  32)
    self.down4 = conv( 64, 128) # ( 64,  16,  16) --> (128,  16,  16)
    self.down5 = conv(128, 256) # (128,   8,   8) --> (256,   8,   8)
    self.up1   = conv(384, 128) # (256,  16,  16) --> (128,  16,  16)
    self.up2   = conv(192,  64) # (128,  32,  32) --> ( 64,  32,  32)
    self.up3   = conv( 96,  32) # ( 64,  64,  64) --> ( 32,  64,  64)
    self.up4   = conv( 48,  16) # ( 32, 128, 128) --> ( 16, 128, 128)
    self.tail  = nn.Conv2d(16, 1, 1)
    self.downpool = nn.MaxPool2d(kernel_size=2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

  def forward(self, x):
    x_down_128 = self.down1(x)
    x_down_64  = self.down2(self.downpool(x_down_128))
    x_down_32  = self.down3(self.downpool(x_down_64))
    x_down_16  = self.down4(self.downpool(x_down_32))
    x_down_8   = self.down5(self.downpool(x_down_16))
    x_up = self.up1(torch.cat([self.upsample(x_down_8), x_down_16], dim=1))
    x_up = self.up2(torch.cat([self.upsample(x_up), x_down_32], dim=1))
    x_up = self.up3(torch.cat([self.upsample(x_up), x_down_64], dim=1))
    x_up = self.up4(torch.cat([self.upsample(x_up), x_down_128],  dim=1))
    return self.tail(x_up)