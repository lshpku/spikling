#
# Dataset and model related tools
#
import os
import re
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image


class SpikeDataset(Dataset):
    '''
    (`spike_sequence`, `ground_truth`) pairs.\n
    Each value has been normalized to (`-1`, `1`), and each
        tensor is of shape (`channel`, `height`, `width`).
    '''

    def __init__(self, path: str):
        self.path = path
        files = os.listdir(self.path)
        self.files = [f for f in files if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = os.path.join(self.path, self.files[idx])
        seq, tag = load_spike_numpy(path)

        # Random rotate
        degree = random.randint(0, 3)
        seq = np.rot90(seq, degree, (1, 2))
        tag = np.rot90(tag, degree, (0, 1))

        # Random fliplr
        if random.random() > 0.5:
            seq = np.flip(seq, 2)
            tag = np.flip(tag, 1)

        seq = seq.astype(np.float) * 2 - 1
        tag = tag.astype(np.float) / 128 - 1
        seq = torch.FloatTensor(seq)
        tag = torch.FloatTensor(tag[np.newaxis, :, :])
        return (seq, tag)


def load_spike_numpy(path: str) -> (np.ndarray, np.ndarray):
    '''
    Load a spike sequence with it's tag from prepacked `.npz` file.\n
    The sequence is of shape (`length`, `height`, `width`) and tag of
        shape (`height`, `width`).
    '''
    data = np.load(path)
    seq, tag, length = data['seq'], data['tag'], int(data['length'])
    seq = np.array([(seq[i // 8] >> (i & 7)) & 1 for i in range(length)])
    return seq, tag


def dump_spike_numpy(path: str, seq: np.ndarray, tag: np.ndarray):
    '''
    Store a spike sequence with it's tag to `.npz` file.
    '''
    length = seq.shape[0]
    seq = seq.astype(np.bool)
    seq = np.array([seq[i] << (i & 7) for i in range(length)])
    seq = np.array([np.sum(seq[i: min(i+8, length)], axis=0)
                    for i in range(0, length, 8)]).astype(np.uint8)
    np.savez(path, seq=seq, tag=tag, length=np.array(length))


def load_spike_raw(path: str, width=400, height=250) -> np.ndarray:
    '''
    Load bit-compact raw spike data into an ndarray of shape
        (`frame number`, `height`, `width`).
    '''
    with open(path, 'rb') as f:
        fbytes = f.read()
    fnum = (len(fbytes) * 8) // (width * height)  # number of frames
    frames = np.frombuffer(fbytes, dtype=np.uint8)
    frames = np.array([frames & (1 << i) for i in range(8)])
    frames = frames.astype(np.bool).astype(np.uint8)
    frames = frames.transpose(1, 0).reshape(fnum, height, width)
    frames = np.flip(frames, 1)
    return frames


def get_latest_version(root: str, model: str) -> int:
    pattern = re.compile('^spikling\\-{}-(\\d{{4}}).pth$'.format(model))
    same = [0]
    for f in os.listdir(root):
        match = pattern.match(f)
        if match:
            same.append(int(match.group(1)))
    return max(same)


def online_generate(model: nn.Module, seq: np.ndarray,
                    device: torch.device, path: str):
    height, width = seq.shape[1:]
    seq = seq[np.newaxis, :, :, :] * 2.0 - 1
    seq = torch.FloatTensor(seq).to(device)
    with torch.no_grad():
        img = model(seq)
    img = np.array(img.to(torch.device('cpu')))
    img = img.reshape(height, width) * 128 + 127
    img = (img * (img >= 0)).astype(np.uint8)
    Image.fromarray(img).save(path)


def online_eval(model: nn.Module, device: torch.device, epoch: int):
    simu_set = ['data/ac0009.npz', 'data/ac0081.npz',
                'eval/ac0071.npz', 'eval/mm0071.npz']
    real_set = ['100kmcar.dat', 'disk-pku_short.dat',
                'number-rotation_short.dat', 'operacut.dat']
    for i in real_set:
        break
        seq = load_spike_raw(os.path.join('eval', i))
        seq = seq[seq.shape[0]//2-16:seq.shape[0]//2+16]
        online_generate(model, seq, device,
                        '{:04d}-{}.png'.format(epoch, i))
    for i, j in enumerate(simu_set):
        seq, _ = load_spike_numpy(j)
        online_generate(model, seq, device,
                        '{:04d}-{}.png'.format(epoch, i))
