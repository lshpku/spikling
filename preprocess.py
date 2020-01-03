#
# Video to Spike Convertor
#
# Divide every frame to 8 sub-frames, as well as pixel values.
# Any pixel >=255 gives out an 1 then is substracted by 255.
#
import cv2
import numpy as np
import os
from utils import dump_spike_numpy


def video_to_spike(path: str, size: tuple, window: int):
    '''
    Transform a video to simulative raw spike data.\n
    Outputs are `.npz` files each containing a pair of array `x` and
        `y`, where `x` is of shape (`window`, `height`, `width`) and
        `y` is of shape (`height`, `width`).
    '''
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    basename = filename[:filename.rfind('.')]
    npzname = '{}-{{:03d}}.npz'.format(basename)
    npzname = os.path.join(dirname, npzname)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('width : {}\nheight: {}\nfps   : {}\nframes: {}'
          .format(width, height, fps, frame_num))

    cratio = min(width/size[0], height/size[1])
    cleft = round((width - size[0]*cratio) / 2)
    cright = width - cleft
    ctop = round((height - size[1]*cratio) / 2)
    cbottom = height - ctop

    adder = np.random.random((size[1], size[0])) * 255
    win_num = 1
    win_size = 0
    xframes = []  # (window, height, width)
    yframe = None   # (height, width)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[ctop:cbottom, cleft:cright, :]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        adder += frame
        saturated = adder >= 255  # saturated is a 0/1 matrix
        xframes.append(saturated)
        adder -= saturated * 255  # or adder *= 1 - saturated ?

        if win_size == window // 2:  # choose the middle frame as y
            yframe = frame
        win_size += 1
        if win_size == window:  # end this window
            xframes = np.array(xframes).astype(np.bool)
            yframe = yframe.astype(np.uint8)
            dump_spike_numpy(npzname.format(win_num), xframes, yframe)
            xframes = []
            win_size = 0
            win_num += 1
            print('\rtransforming: {}/{}'.format(win_num, frame_num/window),
                  end='', flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    files = os.listdir()
    for i in files:
        if i.lower.endswith('.MOV'):
            video_to_spike(i, (128, 128), 32)
