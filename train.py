import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import Generator, Discriminator, VGGLoss
from utils import SpikeDataset, load_model, save_model
from utils import online_eval

LAMBDA_VGG = 1.0
LAMBDA_PIX = 100.0
BATCH_SIZE = 1
MODEL_NAME = 'dd'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG, netD = load_model('.', MODEL_NAME, device=device)

# Note: for BCE-kind losses, labels must be put second!
blgloss = nn.BCEWithLogitsLoss()
l2loss = nn.MSELoss()
vggloss = VGGLoss().to(device)

lr = 0.0002
betas = (0.5, 0.999)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)

dataset = SpikeDataset('data')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=8, pin_memory=True)
evalset = SpikeDataset('eval')
evalloader = DataLoader(evalset, batch_size=BATCH_SIZE, num_workers=8,
                        pin_memory=True)


def train_epoch():
    avg_loss_l2, avg_loss_gan, avg_loss_vgg, avg_loss_D = 0, 0, 0, 0
    size = len(dataloader)
    slen = len(str(size))

    for i, data in enumerate(dataloader):
        x, real_y = data[0].to(device), data[1].to(device)

        '''
        Updata generator
        loss_G = l2_loss + gan_loss + vgg_loss
        '''
        optimizerG.zero_grad()
        fake_y = netG(x)
        loss_vgg = vggloss(real_y, fake_y) * LAMBDA_VGG
        loss_l2 = l2loss(real_y, fake_y) * LAMBDA_PIX

        loss_G = loss_l2 + loss_vgg
        loss_G.backward()
        optimizerG.step()

        avg_loss_l2 += loss_l2.item()
        avg_loss_vgg += loss_vgg.item()
        print('\r{}/{}'.format(str(i+1).rjust(slen), size),
              end='', flush=True)

        if i % 30 == 29:
            eval_loss_vgg, eval_loss_l2 = 0, 0
            for j, data in enumerate(evalloader):
                x, real_y = data[0].to(device), data[1].to(device)
                with torch.no_grad():
                    fake_y = netG(x)
                    eval_loss_vgg += vggloss(real_y, fake_y).item()
                    eval_loss_l2 += l2loss(real_y, fake_y).item()
            eval_loss_vgg *= LAMBDA_VGG
            eval_loss_l2 *= LAMBDA_PIX
            with open('LGVD_eLV.txt', 'a') as f:
                f.write('{}\t{}\t{}\t{}\n'.format(
                    avg_loss_l2/30, avg_loss_vgg/30,
                    eval_loss_l2/len(evalloader), eval_loss_vgg/len(evalloader)))
            avg_loss_l2, avg_loss_gan, avg_loss_vgg, avg_loss_D = 0, 0, 0, 0

    print()


for i in range(50):
    print('epoch {}:'.format(i+1))
    train_epoch()
    save_model(netG, netD, '.', MODEL_NAME)
    online_eval(netG, device, i+1)
