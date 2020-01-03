#
# Display videos that compare each method
#
import torch
from torch import nn
from collections import OrderedDict
from utils import load_model, load_spike_raw
from evaluate import generate, window_method, interval_method
import cv2
import numpy as np


def transform_gen(model: nn.Module, path: str, stride: int):
    seq = load_spike_raw(path)
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output-gen.avi', fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    seq = np.pad(seq, ((0, 0), (1, 1), (0, 0)), 'constant')
    amount=160
    for i in range(amount):
        result = generate(model, seq, i * stride).astype(np.uint8)
        out.write(result)
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()


def transform_raw(path: str, stride: int):
    seq = load_spike_raw(path).astype(np.uint8) * 255
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output-raw.avi', fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    for i in range(amount):
        out.write(seq[i * stride])
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()


def transform_window(path: str, window: int, stride: int):
    seq = load_spike_raw(path).astype(np.uint8)
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output-win.avi', fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    for i in range(amount):
        result = window_method(seq, i * stride, window)
        out.write(result)
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()


def transform_interval(path: str, stride: int):
    seq = load_spike_raw(path).astype(np.uint8)
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output-intv.avi', fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    for i in range(amount):
        result = interval_method(seq, i * stride)
        out.write(result)
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()

PATH = 'test/100kmcar.dat'

netG, _ = load_model('checkpoint', 'dd', 27)
#transform_gen(netG, PATH, 2)
transform_raw(PATH, 2)
transform_window(PATH, 32, 2)
transform_interval(PATH, 2)
