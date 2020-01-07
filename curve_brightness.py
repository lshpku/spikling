import numpy as np
from PIL import Image
from utils import load_spike_raw
from filtering import smoothing, denoising


def brightness_polynome(seq: np.ndarray, k: int) -> np.ndarray:
    nframe, height, width = seq.shape
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            for l0 in range(k-1, -1, -1):
                if seq[l0, i, j] == 1:
                    break
            for l1 in range(l0-1, -2, -1):
                if l0 == -1 or seq[l1, i, j] == 1:
                    break
            for r0 in range(k, nframe):
                if seq[r0, i, j] == 1:
                    break
            for r1 in range(r0+1, nframe+1):
                if r1 == nframe or seq[r1, i, j] == 1:
                    break
            x = [(l1+l0)/2, (l0+r0)/2, (r0+r1)/2]
            y = [255/(l0-l1), 255/(r0-l0), 255/(r1-r0)]
            w = np.polyfit(x, y, 1)
            result[i, j] = min(255, np.polyval(w, k) + 30)
    return result


if __name__ == '__main__':
    pre_frame = load_spike_raw(os.path.join('raw', 'operacut.dat'))
    Image.fromarray(brightness_polynome(pre_frame, 200)).show()
