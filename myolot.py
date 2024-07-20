import math
import os

import numpy as np
import torch

from common_utils.utils import tfOrtModelRuner
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import SpAMt, SCDown, C2ft, SPPFt, PSAt, C2fCIBt, SCUp
from torch import nn

class v10DetectTiny(nn.Module):
    max_det = -1

    def __init__(self, nc=80, ch=(), reg_max=17,use_taa=False):
        super(v10DetectTiny, self).__init__()
        # c = ch[1]
        # c=ch
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        # self.conv = Conv(sum(ch), c, 1)
        self.nc = nc
        self.nl = len(ch)
        self.cv1 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, 1, 1)) for i, x in enumerate(ch))
        self.cv2 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, x, 1)),
                                               nn.Sequential(Conv(x, x, 3, g=x), Conv(x, x, 1)),
                                               nn.Conv2d(x, 4 * self.reg_max, 1)) for i, x in enumerate(ch))
        self.cv3 = nn.ModuleList(nn.Sequential(
            nn.Sequential(Conv(x, x, 3, g=x), Conv(x, x, 1)),
            nn.Conv2d(x, nc, 1)) for i, x in enumerate(ch))
        self.bias_init()
        self.export=False
        self.use_taa=use_taa


    def forward_feat_(self, x, cv1, cv2, cv3):
        y = []
        for i in range(self.nl):
            if self.export:
                if self.use_taa:
                    pcls=cv3[i](x[i]).sigmoid()
                else:
                    pcls = cv3[i](x[i]).softmax(1)
                if self.reg_max==1:
                    xywh=cv2[i](x[i])
                    xy=xywh[:,:2,:,:].tanh()
                    wh=xywh[:,2:,:,:].sigmoid()
                    if self.use_taa or pcls.shape[1]>1:
                        y_ = torch.cat((cv1[i](x[i]).sigmoid(),xy,wh, pcls), 1)
                    else:
                        y_ = torch.cat((cv1[i](x[i]).sigmoid(), xy, wh), 1)

                else:
                    box=cv2[i](x[i])
                    b, _, a = box.shape
                    box=box.view(b, 4, self.reg_max, a).softmax(2).view(b,-1,a)
                    if self.use_taa or pcls.shape[1]>1:
                        y_=torch.cat((cv1[i](x[i]).sigmoid(),box,pcls), 1)
                    else:
                        y_ = torch.cat((cv1[i](x[i]).sigmoid(), box), 1)
                bs, ch, h, w = y_.shape
                y_ = y_.view(bs, ch, h * w)
                y.append(y_)
            else:
                y.append(torch.cat((cv1[i](x[i]).sigmoid(), cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def forward(self, x):
        # x = [self.conv(torch.cat((self.avg_pool(x[0]), x[1], self.upsample(x[2])), 1))]
        x = self.forward_feat_(x, self.cv1, self.cv2, self.cv3)
        # x = torch.cat(x, 2)
        return x

    def bias_init(self):
        # super().bias_init()
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        i = 0
        for a, b in zip(m.cv2, m.cv3):  # from
            # a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (2 ** (i + 3)) ** 2)  # cls (.01 objects, 80 classes, 640 img)



class channelShuffle(nn.Module):
    def __init__(self):
        super(channelShuffle, self).__init__()

    def forward(self,x):
        return torch.cat((x[:, 0::2, :, :], x[:, 1::2, :, :]),dim=1)

class YOLOv10t(nn.Module):
    def __init__(self, nc=80, separation=0, separation_scale=2, reg_max=17,use_taa=False):
        super(YOLOv10t, self).__init__()

        self.separation = separation
        self.separation_scale = separation_scale
        self.channels=[24,32,48,96,192]

        self.stage_list = nn.ModuleList([
            nn.Sequential(
                Conv(3, self.channels[0], 3, 2),
            ),
            nn.Sequential(
                SCDown(self.channels[0], self.channels[1], 3, 2),
                C2ft(self.channels[1], self.channels[1], 1, True),
            ),
            nn.Sequential(
                SCDown(self.channels[1], self.channels[2], 3, 2),
                C2ft(self.channels[2], self.channels[2], 1, True),
            ),
            nn.Sequential(
                SCDown(self.channels[2], self.channels[3], 3, 2),
                C2ft(self.channels[3], self.channels[3], 2, True),
            ),
            nn.Sequential(
                SCDown(self.channels[3], self.channels[4], 3, 2),
                C2ft(self.channels[4], self.channels[4], 1, True),
            ),
        ]
        )

        if self.separation > 0:
            self.stage_list[0].insert(0, SpAMt(0, self.separation,self.separation_scale))
            self.stage_list[self.separation - 1].append(SpAMt(1, self.separation,self.separation_scale))
            self.spa = SpAMt(2, self.separation,self.separation_scale)
        else:
            self.spa = nn.Identity()

        self.spp = nn.Sequential(
            SPPFt(self.channels[4], self.channels[4], 3),
            channelShuffle(),
            PSAt(self.channels[4], self.channels[4]),
        )

        self.s1 = SCUp(self.channels[4], self.channels[3], 3, 2)
        self.conv1 = nn.Sequential(
            # channelShuffle(),
            C2ft(self.channels[3], self.channels[3], 1, True),
        )

        self.s2= SCDown(self.channels[2], self.channels[3], 3, 2)
        self.conv2 = nn.Sequential(
            # channelShuffle(),
            C2fCIBt(self.channels[3], self.channels[3], 1, True, True)
        )

        self.detect = v10DetectTiny(nc, [self.channels[3]], reg_max=reg_max,use_taa=use_taa)

    def forward(self, x):
        x = self.stage_list[1](self.stage_list[0](x))
        p1 = self.stage_list[2](x)
        p2 = self.stage_list[3](p1)
        p3 = self.stage_list[4](p2)
        p3=self.spp(p3)
        p2 = self.conv1(p2 + self.s1(p3))
        x = self.conv2(p2 + self.s2(p1))
        x = self.detect([x])
        x = self.spa(x)
        return x



