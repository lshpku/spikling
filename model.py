import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    '''
    Basic model class that can be saved and loaded
        with specified names.
    '''

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.device = torch.device('cpu')

    def save(self, path):
        torch.save(self.state_dict(), path)
        print('save model to \"{}\"'.format(path))

    def load(self, path: str):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
            print('load pre-trained model \"{}\"'.format(path))
        else:
            print('init model \"{}\"'.format(self.name))
        return self
    
    def to(self, device):
        self.device = device
        self.to(device)


class Generator(BasicModel):
    '''
    Input a (`batch`, `window`, `height`, `width`) sample,
        outputs a (`batch`, `1`, `height`, `width`) result.
    '''

    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.bottom = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.Tanh(),
        )
        self.flat = nn.Conv2d(32, 1, 1, bias=False)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.bottom(d2)
        d2 = self.up1(d2 + d3)
        d1 = self.up2(d1 + d2)
        x = self.flat(x + d1)
        return x


class VGGLoss(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.load_state_dict(torch.load(path))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, real_y, fake_y):
        real_y = real_y.repeat((1, 3, 1, 1))
        fake_y = fake_y.repeat((1, 3, 1, 1))
        with torch.no_grad():
            real_f = self.features(real_y)
        fake_f = self.features(fake_y)
        return F.mse_loss(real_f, fake_f)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == '__main__':
    Generator().save('gen-test.pth')
    Generator().load('gen-test.pth')
    VGGLoss('vgg19-conv.pth')
