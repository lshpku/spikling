#
# Display videos that compare each method
#
import torch
from torch import nn
from collections import OrderedDict
from utils import load_spike_raw
from evaluate import generate, window_method, interval_method
import cv2
import numpy as np


def transform_raw(seq: np.ndarray, stride: int, save_path: str):
    seq = seq.astype(np.uint8) * 255
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    for i in range(amount):
        out.write(seq[i * stride])
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()


def transform_interval(seq: np.ndarray, stride: int, save_path: str):
    seq = seq.astype(np.uint8)
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    for i in range(amount):
        result = interval_method(seq, i * stride)
        out.write(result)
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()


def transform_window(seq: np.ndarray, window: int, stride: int,
                     save_path: str):
    seq = seq.astype(np.float)
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    for i in range(amount):
        result = window_method(seq, i * stride, window)
        out.write(result)
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()


def transform_gen(model: nn.Module, seq: np.ndarray, stride: int,
                  save_path: str):
    seq = seq.astype(np.float)
    length, height, width = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height), 0)
    amount = (length - 32) // stride
    for i in range(amount):
        result = generate(model, seq, i * stride).astype(np.uint8)
        out.write(result)
        print('\rtransforming: {}/{}'.format(i+1, amount), end='')
    print()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
