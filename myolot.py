import math
import os

import numpy as np
import torch

from common_utils.utils import tfOrtModelRuner
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import SpAMt, Bottleneckt, CIB
from torch import nn

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=7 // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = list(torch.chunk(x,4,dim=1))
        return torch.cat((x[0],self.m1(x[1]),self.m2(x[2]),self.m3(x[3])), 1)

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()

        self.cv1 = Conv(c1, c1, k=k, s=s, g=c1, act=False)
        self.cv2 = Conv(c1, c2, 1, 1,act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class SCUp(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1,act=False)
        self.cv2 = nn.Upsample(scale_factor=2,mode='nearest')

    def forward(self, x):
        return self.cv2(self.cv1(x))

class Bottleneckt(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c1, k[0], 1, g=g)
        self.cv2 = Conv(c1, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2ft(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c1=c1
        self.m = nn.ModuleList(
            Bottleneckt(c1 , c1 // 2 ** (1 + _), shortcut, c1 // 2 ** (1 + _), k=((3, 3), (1, 1)), e=1.0)
            for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        for i,m in enumerate(self.m):
            y=m(x)
            x=torch.cat((x[:,:self.c1-self.c1//2**(1+i),:,:],x[:,self.c1-self.c1//2**(1+i):,:,:]+y),dim=1)
        return x


class C2fCIBt(C2ft):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(c1 , c1 // 2 ** (1 + _), shortcut, e=0.5, lk=lk) for _ in range(n))

class v10DetectTiny(nn.Module):
    max_det = -1

    def __init__(self, nc=80, ch=(), reg_max=17):
        super(v10DetectTiny, self).__init__()
        # c = ch[1]
        # c=ch
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        # self.conv = Conv(sum(ch), c, 1)
        self.nc = nc
        self.nl = len(ch)
        self.cv1 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, 1, 1)) for i, x in enumerate(ch))
        self.cv2 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 5, g=x), Conv(x, x, 1)),
                                               nn.Sequential(Conv(x, x, 5, g=x), Conv(x, x, 1)),
                                               nn.Conv2d(x, 4 * self.reg_max, 1)) for i, x in enumerate(ch))
        self.cv3 = nn.ModuleList(nn.Sequential(
            nn.Sequential(Conv(x, x, 5, g=x), Conv(x, x, 1)),
            nn.Conv2d(x, nc, 1)) for i, x in enumerate(ch))
        self.bias_init()
        self.export=False


    def forward_feat_(self, x, cv1, cv2, cv3):
        y = []
        for i in range(self.nl):
            if self.export:
                if self.reg_max==1:
                    xywh=cv2[i](x[i])
                    xy=xywh[:,0:2,:,:].tanh()
                    wh=xywh[:,2:,:,:].sigmoid()
                    y_=torch.cat((cv1[i](x[i]).sigmoid(),xy ,wh, cv3[i](x[i]).softmax(1)), 1)

                else:
                    box=cv2[i](x[i])
                    b, _, a = box.shape
                    box=box.view(b, 4, self.reg_max, a).softmax(2).view(b,-1,a)
                    y_=torch.cat((
                        cv1[i](x[i]).sigmoid(),
                        box,
                        cv3[i](x[i]).softmax(1)), 1)
                bs, ch, h, w = y_.shape
                y_ = y_.view(bs, ch, h * w)
                y.append(y_)
            else:
                y.append(torch.cat((cv1[i](x[i]).sigmoid(), cv2[i](x[i]), cv3[i](x[i]).softmax(1)), 1))

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

# class SPPF(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
#
#     def __init__(self, c1, c2, k=5):
#         """
#         Initializes the SPPF layer with given input/output channels and kernel size.
#
#         This module is equivalent to SPP(k=(5, 9, 13)).
#         """
#         super().__init__()
#         self.c1=c1
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#
#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         for i in range(0,3):
#             y=self.m(x[:, self.c1 - self.c1 // 2 ** (1 + i):, :, :])
#             x=torch.cat((x[:, :self.c1 - self.c1 // 2 ** (1 + i), :, :],x[:, self.c1 - self.c1 // 2 ** (1 + i):, :, :]+y),dim=1)
#         return x

class channelShuffle(nn.Module):
    def __init__(self):
        super(channelShuffle, self).__init__()

    def forward(self,x):
        return torch.cat((x[:, 0::2, :, :], x[:, 1::2, :, :]),dim=1)

class YOLOv10OrtTf():
    def __init__(self, cfg, model_root):
        name_list = os.listdir(model_root)
        model_front_path = None
        model_post_path = None
        model_path = None
        self.separation = cfg.separation
        self.separation_scale = cfg.separation_scale
        for name in name_list:
            if name.endswith(".tflite") or name.endswith(".onnx"):
                if "front" in name:
                    model_front_path = os.path.join(model_root, name)
                elif "post" in name:
                    model_post_path = os.path.join(model_root, name)
                else:
                    model_path = os.path.join(model_root, name)
        if model_path is None:
            assert os.path.splitext(model_front_path)[-1] == os.path.splitext(model_post_path)[-1]
            self.model_type = os.path.splitext(model_front_path)[-1]
            self.model_front = tfOrtModelRuner(model_front_path)
            self.model_post = tfOrtModelRuner(model_post_path)
            if model_front_path.endswith(".tflite"):
                std0,mean0=self.model_front.model_output_details[0]["quantization"]
                std1,mean1=self.model_post.model_input_details["quantization"]
                self.fix0 = std0/std1
                self.fix1 = -self.fix0*mean0+mean1
                self.weight, self.bias = self.model_front.model_input_details["quantization"]
            self.sp = 1
        else:
            self.model = tfOrtModelRuner(model_path)
            self.sp = 0
            self.model_type = os.path.splitext(model_path)[-1]
            self.weight, self.bias = self.model.model_input_details["quantization"]

    def __call__(self, inputs):
        pred_list = []
        for x in inputs:
            if self.model_type == ".tflite":
                x = x.permute(1, 2, 0).cpu().numpy()
                x = x.astype("float32")
                x=x/self.weight+self.bias
                x=np.clip(x,-128,127)
                x=x.astype("int8")
                h, w, c = x.shape[:3]
            else:
                x = x.cpu().numpy()
                c,h,w = x.shape[:3]
            h0 = h
            w0 = w
            if self.sp == 1:
                y_list = []
                for r in range(0, self.separation_scale):
                    for c in range(0, self.separation_scale):
                        if self.model_type==".tflite":
                            x_ = x[None, r * h // self.separation_scale:(r + 1) * h // self.separation_scale,
                                 c * w // self.separation_scale:(c + 1) * w // self.separation_scale, :]
                        else:
                            x_ = x[None, :,r * h // self.separation_scale:(r + 1) * h // self.separation_scale,
                                 c * w // self.separation_scale:(c + 1) * w // self.separation_scale]
                        y = self.model_front(x_)[0]
                        y_list.append(y)
                if self.model_type==".tflite":
                    h, w, c = y_list[0].shape[:3]
                    y = np.zeros((h * self.separation_scale, w * self.separation_scale, c), dtype="int8")
                else:
                    c,h,w = y_list[0].shape[:3]
                    y = np.zeros((c,h * self.separation_scale, w * self.separation_scale), dtype="float32")
                id = 0
                for r in range(0, self.separation_scale):
                    for c in range(0, self.separation_scale):
                        if self.model_type == ".tflite":
                            y[ r * h:(r + 1) * h, c * w:(c + 1) * w,:] = y_list[id]
                        else:
                            y[:,r * h:(r + 1) * h, c * w:(c + 1) * w] = y_list[id]
                        id += 1
                y = y[None, :, :, :].astype('float32')
                if self.model_type == ".tflite":
                    y = np.clip(y*self.fix0+self.fix1,-128,127)
                    y = y.astype('int8')
                out = self.model_post(y)
            else:
                out = self.model(x[None, :, :, :])
                # out = np.concatenate(out,axis=2)
                # pred_list0.append(out)
            if self.model_type == ".tflite":
                out = torch.tensor(out, device=inputs.device).permute(0, 2,1)
            else:
                out = torch.tensor(out, device=inputs.device)
            bs,ch,hw=out.shape
            scale=int((h0*w0//hw)**0.5)
            out=out.view(bs,ch,h0//scale,w0//scale)
            pred_list.append(out)
        out = torch.cat(pred_list, dim=0)
        return out

class YOLOv10t(nn.Module):
    def __init__(self, nc=80, separation=0, separation_scale=2, reg_max=17):
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
                C2ft(self.channels[2], self.channels[2], 2, True),
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
            SPPF(self.channels[4], self.channels[4], 3),
            # channelShuffle(),
            # PSAt(self.channels[4], self.channels[4]),
        )

        self.s1 = SCUp(self.channels[4], self.channels[3], 3, 2)
        self.conv1 = nn.Sequential(
            # channelShuffle(),
            C2fCIBt(self.channels[3], self.channels[3], 2, True, True)
        )

        self.s2= SCDown(self.channels[2], self.channels[3], 3, 2)
        self.conv2 = nn.Sequential(
            # channelShuffle(),
            C2fCIBt(self.channels[3], self.channels[3], 2, True, True)
        )

        self.detect = v10DetectTiny(nc, [self.channels[3]], reg_max=reg_max)

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



