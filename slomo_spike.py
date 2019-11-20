#
# Slow Motion to Spike Array Convertor
#
# Divide every frame to 8 sub-frames, as well as pixel values.
# Any pixel >=255 gives out an 1 then is substracted by 255.
#
import cv2
import numpy as np


def slomo2spike(filename: str, display=False, window=4):
    cap = cv2.VideoCapture(FILE_NAME)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    repvideo = cv2.VideoWriter(
        'rpframes.avi', fourcc, 30.0, (width, height), 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('width : {}\nheight: {}\nfps   : {}\nframes: {}'
          .format(width, height, fps, frame_cnt))

    rpframes = []
    frame_adder = np.random.random((height, width)) * 255

    while(True):
        print('\rprocessing frame: {}/{}'.format(
            str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            .rjust(len(str(frame_cnt))), frame_cnt), end='', flush=True)

        ret, frame = cap.read()
        if not ret:
            break

        frame = np.mean(frame, axis=2)
        frame_adder += frame / window
        overflow = frame_adder >= 255  # overflow is a boolean matrix
        rpframes.append(overflow)
        overflow = overflow * 255
        repvideo.write(overflow.astype(np.uint8))
        frame_adder -= overflow

        if display:
            cv2.imshow('overflow', overflow.astype(np.uint8))
            cv2.waitKey(8)

    np.save('rpframes.npy', rpframes)
    cap.release()
    repvideo.release()
    cv2.destroyAllWindows()
    print('\nfinished')


def restore(filename='rpframes.npy', window=32, fps=30.0,
            brightness=4, display=False):
    rpframes = np.load(filename)
    frame_cnt, height, width = rpframes.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    resvideo = cv2.VideoWriter('restore.avi', fourcc, 30.0, (width, height), 0)
    frjust = len(str(frame_cnt))
    print('width : {}\nheight: {}\nframes: {}'
          .format(width, height, frame_cnt))

    frame_addr = np.empty((height, width), dtype=float)
    window = int(window)
    luminance = (256 // window) * brightness
    print('luminace: '.format(luminance))
    cnt = 0

    for idx, frame in enumerate(rpframes):
        print('\rprocessing frame: {}/{}'
              .format(str(idx+1).rjust(frjust), frame_cnt), end='', flush=True)

        frame_addr += frame * luminance
        if idx >= window:
            frame_addr -= rpframes[idx - window] * luminance
        overflow = (frame_addr > 255).astype(np.int)
        frame_disp = frame_addr * (1 - overflow) + overflow * 255
        cnt += 1
        if cnt == 32:
            resvideo.write(frame_disp.astype(np.uint8))
            cnt = 0

        if display:
            cv2.imshow('restore', frame_disp.astype(np.uint8))
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    resvideo.release()
    print('\nfinished')


FILE_NAME = 'steak_comp.mp4'

if __name__ == '__main__':
    slomo2spike(FILE_NAME)
    restore()
