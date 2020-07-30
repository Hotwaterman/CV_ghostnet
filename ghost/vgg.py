import torch
import torch.nn as nn
import math

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

netkind = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'C' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class Vgg(nn.Module):
    def __init__(self, feature, num_class=10):
        super(Vgg, self).__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        self._init_weight()
    
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        out = self.classifier(x)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def _make_layer(netkind, batch_norm=False):
    layers = []
    inputs = 3
    for i in netkind:
        if i == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers.append(nn.Conv2d(inputs, i, kernel_size=3, padding=1))

        if batch_norm:
            layers.append(nn.BatchNorm2d(i))

        layers.append(nn.ReLU(inplace=True))
        inputs = i

    return nn.Sequential(*layers)

def vgg11():
    return Vgg(_make_layer(netkind['A'], batch_norm=True))

def vgg13():
    return Vgg(_make_layer(netkind['B'], batch_norm=True))

def vgg16():
    return Vgg(_make_layer(netkind['C'], batch_norm=True))

def vgg19():
    return Vgg(_make_layer(netkind['D'], batch_norm=True))