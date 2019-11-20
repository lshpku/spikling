#
# A Test on OpenCV's Farneback Optical Flow Method
#
import cv2
import numpy as np
import signal


cap = cv2.VideoCapture('steak_comp.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
repvideo = cv2.VideoWriter(
    'steak_optflow.avi', fourcc, 25.0, (width, height))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

force_break = False  # if you don't want to wait too long
print('Press Ctrl-C to save and break')

def handle_break(signum, frame):
    global force_break
    force_break = True


signal.signal(signal.SIGINT, handle_break)
signal.signal(signal.SIGTERM, handle_break)

for i in range(frame_cnt):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gunnar-Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(
        prevgray, gray, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    prevgray = gray

    # Flow is a matrix of (x, y)s telling each point's shift
    # Here we convert it to (y, Cb, Cr) to visualize
    x_shift, y_shift = flow[:, :, 0], flow[:, :, 1]
    shift = np.sqrt(x_shift**2 + y_shift**2)
    shift = shift + (shift == 0)*1e-128  # avoid div0
    y_chn = (-np.exp(-shift) + 1) * 256
    cb_chn = ((x_shift / shift) + 1)*128
    cr_chn = ((y_shift / shift) + 1)*128

    cframe = np.dstack((y_chn, cb_chn, cr_chn)).astype(np.uint8)
    cframe = cv2.cvtColor(cframe, cv2.COLOR_YCrCb2RGB)

    repvideo.write(cframe.astype(np.uint8))
    print('\rrunning: {}/{}'.format(i, frame_cnt), end='')
    if force_break:
        break

print('\ndone')
cv2.destroyAllWindows()
repvideo.release()
