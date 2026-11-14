"""Small U-Net-style heatmap predictor for center localization (PyTorch)."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1),nn.ReLU(inplace=True),
        )

    def forward(self,x): return self.net(x)


class CenterHeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=ConvBlock(3,16)
        self.enc2=ConvBlock(16,32)
        self.bottleneck=ConvBlock(32,64)
        self.dec2=ConvBlock(64+32,32)
        self.dec1=ConvBlock(32+16,16)
        self.output=nn.Conv2d(16,1,1)

    def forward(self,x):
        a=self.enc1(x)
        b=self.enc2(F.avg_pool2d(a,2))
        c=self.bottleneck(F.avg_pool2d(b,2))
        d=self.dec2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],dim=1))
        e=self.dec1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],dim=1))
        return self.output(e)[:,0]  # (B, H, W) unnormalized center logits


def spatial_probs(logits):
    b,h,w=logits.shape
    return torch.softmax(logits.reshape(b,-1),dim=1).reshape(b,h,w)


def location_from_probs(probs):
    """Argmax pixel (x,y) and its probability; good for sharp heatmaps."""
    h,w=probs.shape
    idx=int(probs.reshape(-1).argmax().item())
    return idx%w,idx//w,float(probs.reshape(-1)[idx].item())
