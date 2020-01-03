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
import matplotlib.pyplot as plt
from utils import load_spike_numpy, load_spike_raw


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
    Generate an image using SpikeCNN.\n
    Heights and widths that are not aligned to 4 will be padded before
        putting into the network, and restored before return.
    '''
    height, width = seq.shape[1:]
    seq = seq.astype(np.float)
    # TODO: adaptative padding
    seq = np.pad(seq, ((0, 0), (1, 1), (0, 0)), mode='constant')
    seq = seq[np.newaxis, start:start+32, :, :] * 2 - 1
    seq = torch.FloatTensor(seq)
    with torch.no_grad():
        result = model(seq)
    # TODO: adaptative cropping
    result = result.cpu().numpy().reshape(height+2, width)
    result = result[1:-1] * 128 + 127
    result = result * ((result <= 255) & (result >= 0)) + 255 * (result > 255)
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


if __name__ == '__main__':
    pass
