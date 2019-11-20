#
# An Interface to Load Spike Camera Raw Data
#
import numpy as np

FILENAME = '100kmcar.dat'


def load_spike_data(path: str, width=400, height=250):
    frame_size = (width * height) // 8
    dat = open(path, 'rb')
    bframes = dat.read()
    frames = []
    frame_cnt = len(bframes) // frame_size
    for i in range(frame_cnt):
        frame = bframes[i*frame_size:(i+1)*frame_size]
        rows = [[]]
        for bt in frame:  # integer in [0, 255]
            if len(rows[-1]) == width:
                rows.append([])
            rows[-1] += [(bt >> k) & 1 for k in range(8)]
        rows = np.array(rows).astype(np.bool)
        frames.append(np.flipud(rows))
        print(i, frame_cnt)
    frames = np.array(frames)
    return frames


if __name__ == '__main__':
    frames = load_spike_data(FILENAME)
    print(frames.shape)
