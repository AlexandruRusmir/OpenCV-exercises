import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('D:/wamp64/www/Prelucrarea Imaginilor/python/imagine.jpeg')

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_grey = cv2.calcHist([grey], [0], None, [256], [0, 256])
hist_color = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure()
plt.title("Histograma imaginii in tonuri de gri")
plt.plot(hist_grey)
plt.xlim([0, 256])

plt.figure()
plt.title("Histograma imaginii color")
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])


def minmax(v):
    if v > 255:
        return 255
    if v < 0:
        return 0
    return v


def floyd_dithering(inMat, samplingF):
    h = inMat.shape[0]
    w = inMat.shape[1]

    for y in range(0, h - 1):
        for x in range(1, w - 1):
            old_p = inMat[y, x]
            new_p = np.round(samplingF * old_p / 255.0) * (255 / samplingF)
            inMat[y, x] = new_p

            quant_error_p = old_p - new_p

            inMat[y, x + 1] = minmax(inMat[y, x + 1] + quant_error_p * 7 / 16.0)
            inMat[y + 1, x - 1] = minmax(inMat[y + 1, x - 1] + quant_error_p * 3 / 16.0)
            inMat[y + 1, x] = minmax(inMat[y + 1, x] + quant_error_p * 5 / 16.0)
            inMat[y + 1, x + 1] = minmax(inMat[y + 1, x + 1] + quant_error_p * 1 / 16.0)

    return inMat


img = cv2.medianBlur(img, 5)
th1 = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Tonuri de gri', grey)
cv2.imshow('Praguri multiple', th1)
floyd_gray = floyd_dithering(grey, 1)
cv2.imshow('Floyd-Steinberg', floyd_gray)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()