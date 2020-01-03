#
# Evaluate with pictures
#
import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import load_spike_numpy, load_spike_raw, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def interval_method(seq: np.ndarray, mid: int) -> np.ndarray:
    '''
    Snapshot an image using interval method.
    '''
    length, height, width = seq.shape
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            up, down = mid, mid-1
            for up in range(mid, length):
                if seq[up, i, j] == 1:
                    break
            for down in range(mid-1, -1, -1):
                if seq[down, i, j] == 1:
                    break
            result[i, j] = 255 / (up - down)
    return result.astype(np.uint8)


def window_method(seq: np.ndarray, start: int, length: int) -> np.ndarray:
    '''
    Generate an image using window method.
    '''
    result = seq[start:start+length].mean(axis=0) * 255
    return result.astype(np.uint8)


def generate(model: nn.Module, seq: np.ndarray, start: int) -> np.ndarray:
    '''
    Generate an image using SpikeCNN.
    '''
    height, width = seq.shape[1:]
    seq = seq.astype(np.float)
    seq = seq[np.newaxis, start:start+32, :, :] * 2 - 1
    seq = torch.FloatTensor(seq)
    with torch.no_grad():
        result = model(seq)
    result = result.numpy().reshape(height, width)
    result = result * 128 + 127
    result = result * ((result <= 255) & (result >= 0)) + 255 * (result > 255)
    img = Image.fromarray(result.astype(np.uint8))
    img.save('temp/{}.png'.format(str(random.random())[2:]))
    return result.astype(np.uint8)


def plot_with_raw(model: nn.Module, raw_path: str, save_path: str):
    '''
    Show a plot of `raw spike`, `interval method`,
        `window method` and `SpikeCNN` results.
    '''
    raw = load_spike_raw(raw_path)[:, 1:249, :]
    mid = raw.shape[0] // 2
    snap = raw[mid, :, :] * 255
    window = window_method(raw, mid-16, 32)
    interval = interval_method(raw, mid)
    result = generate(model, raw, mid-16)

    fig = plt.figure(figsize=(11, 8))
    titles = ['Raw Spike Snapshot', 'Interval Method',
              'Window Method (32)', 'SpikeCNN (32)']
    images = [snap, interval, window, result]
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        ax.set_title(titles[i], fontsize=18)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path)
    plt.close('all')


def plot_with_numpy(model: nn.Module, npz_path: str, save_path: str):
    '''
    Show a plot of `ground truth`, `interval method`,
        `window method` and `SpikeCNN` results.
    '''
    seq, tag = load_spike_numpy(npz_path)
    window = window_method(seq, 0, 32)
    interval = interval_method(seq, 16)
    result = generate(model, seq, 0)

    fig = plt.figure(figsize=(9, 9))
    titles = ['Ground Truth', 'Interval Method',
              'Window Method (32)', 'SpikeCNN (32)']
    images = [tag, interval, window, result]
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        ax.set_title(titles[i], fontsize=18)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(save_path)
    plt.close('all')


def result_path(path: str, model: str, version: int) -> str:
    base = os.path.basename(path)
    suffix_idx = base.rfind('.')
    if suffix_idx != -1:
        base = base[:suffix_idx]
    name = '{}-{}-{:04d}.png'.format(base, model, version)
    return os.path.join('result', name)


def test_with_raw(model: nn.Module, raw_path: str):
    raw = load_spike_raw(raw_path)
    path = os.path.basename(raw_path)
    path = path[:path.rfind('.')]
    mid = raw.shape[0] // 2
    Image.fromarray(raw[mid, :, :] * 255).save('{}-snap.png'.format(path))
    Image.fromarray(window_method(raw, mid-16, 32)).save('{}-win.png'.format(path))
    Image.fromarray(interval_method(raw, mid)).save('{}-intv.png'.format(path))
    raw = np.pad(raw, ((0, 0), (1, 1), (0, 0)), 'constant')
    result = generate(model, raw, mid-16)
    Image.fromarray(result[1:251, :]).save('{}-cnn.png'.format(path))


def test_with_numpy(model: nn.Module, raw_path: str):
    raw, tag = load_spike_numpy(raw_path)
    path = os.path.basename(raw_path)
    path = path[:path.rfind('.')]
    Image.fromarray(tag).save('{}-gt.png'.format(path))
    Image.fromarray(window_method(raw, 0, 32)).save('{}-win.png'.format(path))
    Image.fromarray(interval_method(raw, 16)).save('{}-intv.png'.format(path))
    result = generate(model, raw, 0)
    Image.fromarray(result).save('{}-cnn.png'.format(path))


if __name__ == '__main__':
    real_set = ['100kmcar.dat', 'disk-pku_short.dat',
                'number-rotation_short.dat', 'operacut.dat']
    models = [('dd', i) for i in [10]]
    for m in models:
        netG, _ = load_model('checkpoint', *m)
        for f in real_set:
            #break
            test_with_raw(netG, os.path.join('test', f))

        for f in ['ev0021.npz']:
            #break
            test_with_numpy(netG, os.path.join('eval', f))

        for f in ['ac0405.npz', 'mc0049.npz']:
            break
            plot_with_numpy(netG, os.path.join('data', f),
                            result_path(f, *m))
