import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Generator, VGGLoss
from utils import SpikeDataset

LAMBDA_VGG = 1.0
LAMBDA_L2 = 100.0
BATCH_SIZE = 1
EVAL_EVERY = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = Generator().to(device)

l2loss = nn.MSELoss()
vggloss = VGGLoss('vgg19-conv.pth').to(device)

lr = 0.0002
betas = (0.5, 0.999)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)

dataset = SpikeDataset('data')
dataloader = DataLoader(dataset, BATCH_SIZE, True, pin_memory=True)
evalset = SpikeDataset('eval')
evalloader = DataLoader(evalset, BATCH_SIZE, pin_memory=True)


def train_epoch():
    avg_loss_l2, avg_loss_vgg = 0, 0
    size = len(dataloader)
    slen = len(str(size))

    for i, data in enumerate(dataloader):
        x, real_y = data[0].to(device), data[1].to(device)
        
        optimizerG.zero_grad()
        fake_y = netG(x)
        loss_vgg = vggloss(real_y, fake_y) * LAMBDA_VGG
        loss_l2 = l2loss(real_y, fake_y) * LAMBDA_PIX

        loss_G = loss_l2 + loss_vgg
        loss_G.backward()
        optimizerG.step()

        avg_loss_l2 += loss_l2
        avg_loss_vgg += loss_vgg
        print('\r{}/{}'.format(str(i+1).rjust(slen), size),
              end='', flush=True)

        if (i+1) % EVAL_EVERY == 0:
            eval_loss_vgg, eval_loss_l2 = online_eval()
            avg_loss_l2 = avg_loss_l2.item() / EVAL_EVERY
            avg_loss_vgg = avg_loss_vgg.item() / EVAL_EVERY

            with open('LGVD_eLV.txt', 'a') as f:
                f.write('{}\t{}\t{}\t{}\n'.format(
                    avg_loss_l2, avg_loss_vgg,
                    eval_loss_l2, eval_loss_vgg))
            avg_loss_l2, avg_loss_vgg = 0, 0

    print()


def online_eval():
    eval_loss_l2, eval_loss_vgg = 0, 0

    for j, data in enumerate(evalloader):
        x, real_y = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            fake_y = netG(x)
            eval_loss_l2 += l2loss(real_y, fake_y)
            eval_loss_vgg += vggloss(real_y, fake_y)

    eval_loss_l2 *= LAMBDA_L2 / len(evalloader)
    eval_loss_vgg *= LAMBDA_VGG / len(evalloader)

    return eval_loss_l2.item(), eval_loss_vgg.item()


for i in range(50):
    print('epoch {}:'.format(i+1))
    train_epoch()
    save_model(netG, netD, '.', MODEL_NAME)
    online_eval(netG, device, i+1)
