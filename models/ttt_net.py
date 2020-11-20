import torch
from torch import nn

class MVN(nn.Module):
    def __init__(self, esp=1e-6):
        super(MVN, self).__init__()
        self.esp = esp

    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True)
        x = (x-mean)/(std+self.esp)
        return x


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, filters=3, strides=1, padding=1, norm='mvn', reps=2):
        super(conv_block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.convs = nn.ModuleList()
        in_conv = self.in_c
        for i in range(reps):
            self.convs.append(nn.Conv2d(in_conv, self.out_c,
                                        filters, strides, padding=padding))
            if norm == 'mvn':
                self.convs.append(MVN())
            elif norm == 'bn':
                self.convs.append(nn.BatchNorm2d(self.out_c))
            elif norm == 'mvn+bn':
                self.convs.append(MVN())
                self.convs.append(nn.BatchNorm2d(self.out_c))
            self.convs.append(nn.ReLU(inplace=True))
            in_conv = self.out_c
        self.convs.append(nn.Conv2d(self.out_c, self.out_c, 3, 2, padding=1))

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x


class TTTnet(nn.Module):
    def __init__(self, n_class, n_channel, norm='mvn'):
        super(TTTnet, self).__init__()
        self.n_class = n_class
        self.norm = norm
        self.encoder = nn.ModuleList([
            conv_block(n_channel, 64, norm=self.norm),
            conv_block(64, 128, norm=self.norm),
            conv_block(128, 256, norm=self.norm),
            conv_block(256, 512, norm=self.norm),
        ])
        self.output = nn.Sequential(
            nn.Conv2d(512, self.n_class, 1), nn.Sigmoid())

    def forward(self, x):
        # fms = []
        for i, block in enumerate(self.encoder):
            x = block(x)
        x = self.output(x)
        return x
