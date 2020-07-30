import torch
import torch.nn as nn
import math

__all__=['ghost_net']

class SEmodule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEmodule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #squeeze 
        self.func = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        ) #bottleneck, build the connection between the channels
    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.func(out).view(b, c, 1, 1)
        return x*out

class Ghostmodule(nn.Module):
    def __init__(self, inputs, out, kernel_size=1, s=2, d=3, stride=1):
        super(Ghostmodule, self).__init__()
        in_channel = math.ceil(out/s)
        hid = in_channel*(s-1)

        self.conv = nn.Conv2d(inputs, in_channel, kernel_size, stride, kernel_size//2, bias=False)
    
        self.cheapop = nn.Conv2d(in_channel, hid, d, 1, d//2, groups=in_channel, bias=False)
          
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.cheapop(x1)
        out = torch.cat([x1, x2], 1)
        return out

def DWconv(x, y, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(x, y, kernel_size, stride, kernel_size//2, groups=x, bias=False),
            nn.BatchNorm2d(y)
        )

class Bneck(nn.Module):
    def __init__(self, inputs, hid, out, kernel_size, stride, se):
        super(Bneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            Ghostmodule(inputs, hid, kernel_size=1),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            DWconv(hid, hid, kernel_size, stride) if stride==2 else nn.Sequential(),
            SEmodule(hid) if se else nn.Sequential(),
            Ghostmodule(hid, out, kernel_size=1),
            nn.BatchNorm2d(out)
        )

        if stride == 1 and inputs == out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                DWconv(inputs, inputs, 3, stride),
                nn.ReLU(inplace=True),
                nn.Conv2d(inputs, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

        
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Ghostnet(nn.Module):
    def __init__(self, classnumber=10, wd=1):
        super(Ghostnet, self).__init__()
        self.con3x3 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            Bneck(16, 16, 16, kernel_size=3, stride=1, se=0),
            Bneck(16, 48, 24, kernel_size=3, stride=2, se=0)
        )
        self.block2 = nn.Sequential(
            Bneck(24, 72, 24, kernel_size=3, stride=1, se=0),
            Bneck(24, 72, 40, kernel_size=3, stride=2, se=1)
        )
        self.block3 = nn.Sequential(
            Bneck(40, 120, 40, kernel_size=3, stride=1, se=1),
            Bneck(40, 240, 80, kernel_size=3, stride=2, se=0)
        )
        self.block4 = nn.Sequential(
            Bneck(80, 200, 80, kernel_size=3, stride=1, se=0),
            Bneck(80, 184, 80, kernel_size=3, stride=1, se=0),
            Bneck(80, 184, 80, kernel_size=3, stride=1, se=0),
            Bneck(80, 480, 112, kernel_size=3, stride=1, se=1),
            Bneck(112, 672, 112, kernel_size=3, stride=1, se=1),
            Bneck(112, 672, 160, kernel_size=3, stride=2, se=1)
        )
        self.block5 = nn.Sequential(
            Bneck(160, 960, 160, kernel_size=3, stride=1, se=0),
            Bneck(160, 960, 160, kernel_size=3, stride=1, se=1),
            Bneck(160, 960, 160, kernel_size=3, stride=1, se=0),
            Bneck(160, 960, 160, kernel_size=3, stride=1, se=1),
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(960, 1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(1280, classnumber)

        self.initweight()

    def forward(self, x):
        out = self.con3x3(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.avg_pool(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def initweight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def ghost_net():
    return Ghostnet()

        




