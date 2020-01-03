# spikling
Spike Camera Image Restoration

## Requirements
```
Python 3.7
pytorch>=1.1.0
opencv-python
matplotlib
```

## Network Architecture
We use a Semi-UNet CNN to solve this problem. It's proved that Semi-UNet dose better in denoise and demotion than Full-UNet.
![](/image/network.png)

## Results
We have tested on real and simulative spike sequences, and SpikeCNN reconstructs approximately ground truth when it's bright. However, interval method takes a lead when it's dim. It's because SpikeCNN as well as window method is limited to a fixed window size, so it cannot capture enough light to reconstruct the original scene.<br>
### Real Spikes
![](/image/test-real.png)
### Simulative Spikes
![](/image/test-simu.png)

## 使用方法
本处理系统提供了**上手即用**的脉冲相机数据到图片、视频的转换，并为想要自己制造数据集进行训练的用户提供了数据预处理和训练的代码。<br>
注：(1) 本代码已在带CUDA的Windows和不带CUDA的macOS上通过了测试；(2) 请根据各文件的实际路径调整参数。
### 使用传统方法进行转换
- 加载脉冲相机数据为NumPy数组
```python
import os
from utils import load_spike_raw

seq = load_spike_raw(os.path.join('raw', 'operacut.dat'))
```
- 将特定帧转换为图片
```python
from PIL import Image
from evaluate import interval_method, window_method

# 间隔法
MIDDLE_FRAME = len(seq)//2  # 假设要截取中间的一帧

result = interval_method(seq, MIDDLE_FRAME)
img = Image.fromarray(result)
img.save(os.path.join('result', 'operacut-intv.png'))

# 窗口法
START_FRAME = len(seq)//2
WINDOW_SIZE = 32

result = window_method(seq, START_FRAME, WINDOW_SIZE)
img = Image.fromarray(result)
img.save(os.path.join('result', 'operacut-win.png'))
```
- 从脉冲序列生成视频
```python
from display import (
    transform_raw,
    transform_interval,
    transfrom_window,
)

STRIDE = 2  # 每生成1帧视频跨越2帧spike数据

# 由于原始序列太长，这里只转换一小部分
seq_s = seq[:48]

# 展示原始脉冲数据
transform_raw(seq_s, STRIDE,
              os.path.join('result', 'operacut-raw.avi'))

# 间隔法
transform_interval(seq_s, STRIDE,
                   os.path.join('result', 'operacut-intv.avi'))

# 窗口法
transform_window(seq_s, WINDOW_SIZE, STRIDE,
                 os.path.join('result', 'operacut-win.avi'))
```
### 使用深度学习方法进行转换
- 加载预训练模型
```python
import os
import torch
from model import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Generator()
gen.load(os.path.join('checkpoint', 'spikling-0027.pth'))
```
- 将特定帧转换为图片
```python
from PIL import Image
from utils import load_spike_raw
from evaluate import generate

seq = load_spike_raw(os.path.join('raw', 'operacut.dat'))

START_FRAME = len(seq)//2

result = generate(gen, seq, START_FRAME)
img = Image.fromarray(result)
img.save(os.path.join('result', 'operacut-gen.png'))
```
- 从脉冲序列生成视频（建议在GPU环境下进行）
```python
from display import transform_gen

STRIDE = 2

seq_s = seq[:48]

transform_gen(gen, seq_s, STRIDE,
              os.path.join('result', 'operacut-gen.avi'))
```
### 训练深度学习模型
- 生成模拟脉冲数据<br>

```python
import os
from preprocess import video_to_spike

# 必须与当前SpikeCNN的设置一致
WINDOW_SIZE = 32
FRAME_SIZE = (256, 256)

my_videos = os.listdir('video')  # 源视频存放的位置
out_pattern = 'sp{:04d}.npz'  # 结果命名模板

for i in my_videos:
    if i.lower.endswith('.mp4'):
        out_name = os.path.join('data', out_pattern)
        video_to_spike(i, FRAME_SIZE, WINDOW_SIZE, out_name)
```
- 训练

我们已经提供了一套调好的超参数，如果您必须调整，您需要在`train.py`中设置，可设置的参数有：
```python
BATCH_SIZE = 16
EVAL_EVERY = 30  # 每30个batch在验证集上跑一次

# VGGLoss和L2Loss占的比例
LAMBDA_VGG = 1.0
LAMBDA_L2 = 100.0

# Adam优化器的超参数
lr = 0.0002
betas = (0.5, 0.999)

# 以及trainset和evalset的路径
```
为了实现VGGLoss，您还需要一个预训练好的VGG19模型，您可以轻松在其他人的项目中下载此类模型，故在此不提供。<br>
运行`train.py`开始训练（建议在GPU环境下进行）
```bash
$ python3 train.py
```
### 预训练模型与预处理数据集
我们提供了一个预训练的模型权重`checkpoint/spikling-0027.pth`，以及一个含有5.2k段模拟脉冲序列的数据集`data/`，下载地址：[北大网盘](https://disk.pku.edu.cn:443/link/B859EF922D2EAEA5AEA9EC1415DDA103 "北大网盘")；或：[百度网盘](https://pan.baidu.com/s/1JnzcsHROTUvHu6T8EyY6EQ "百度网盘")（密码：x23f）。<br>
我们没有对训练集与验证集进行划分，您可以自行划分，例如用下面的代码将1%的数据*随机*划分到验证集：
```bash
$ mv ./data/*01.pth ./eval
```
真实脉冲相机的数据属于学校资源，因版权问题无法提供，但您仍然可以用模拟脉冲数据体会该模型的神奇之处。
