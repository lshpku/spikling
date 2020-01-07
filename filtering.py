import numpy as np

def denoising(a: np.ndarray) -> np.ndarray:
    b = np.zeros((height,width))
    dx = [0,1,1,1,0,-1,-1,-1]
    dy = [-1,-1,0,1,1,1,0,-1]
    for i in range(1, height-1):
        for j in range(1, width -1):
            t = 0
            for k in range(8):
                if(abs(a[i][j]-a[i+dx[k]][j+dy[k]])>20): t+=1
            if(t > 5):
                for k in range(8):
                        b[i][j] += a[i+dx[k]][j+dy[k]]
                a[i][j] = (a[i][j]+ b[i][j]/2) / 5
    return a


def smoothing(a: np.ndarray) -> np.ndarray:
    b = np.zeros((height,width))
    c = np.zeros((height,width))
    L,w = 20,0.5
    for i in range(height):
        for j in range(width):
            for k in range(1,3):
                for x in range(0,k+1):
                    y = k - x
                    if(i + x < height and j + y < width and abs(a[i][j] - a[i + x][j + y]) < L):
                        b[i][j] += a[i + x][j + y] * (w - k * 0.1),
                        c[i][j] += (w - k * 0.1)
                    if(i - x >= 0 and j + y < width and abs(a[i][j] - a[i - x][j + y]) < L):
                        b[i][j] += a[i - x][j + y] * (w - k * 0.1),
                        c[i][j] += (w - k * 0.1)
                    if(i + x < height and j - y >= 0 and abs(a[i][j] - a[i + x][j - y]) < L):
                        b[i][j] += a[i + x][j - y] * (w - k * 0.1),
                        c[i][j] += (w - k * 0.1)
                    if(i - x >= 0  and j - y >= 0 and abs(a[i][j] - a[i - x][j - y]) < L):
                        b[i][j] += a[i - x][j - y] * (w - k * 0.1),
                        c[i][j] += (w - k * 0.1)
            b[i][j] = (a[i][j]+b[i][j])/(c[i][j]+1)
    fw = open("disk-pku_short.txt", "w")
    for i in range(height):
        for j in range(200):
            fw.write(str(int(b[i, j]))+" ")
        fw.write("\n")
    return b
