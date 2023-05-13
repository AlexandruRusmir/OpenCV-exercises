import cv2
import numpy as np
from scipy.ndimage.filters import convolve

# Functie pentru a aplica binarizarea adaptiva
def binarizare_adaptiva(img, blockSize, C):
    # Converteste imaginea in nivele de gri
    img_gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplica filtrul media (blur) pentru a obtine imaginea medie in blocuri
    img_mean = cv2.blur(img_gri, (blockSize, blockSize))
    # Binarizeaza imaginea pe baza valorilor medii locale
    img_bin = np.where(img_gri > img_mean - C, 255, 0).astype(np.uint8)
    return img_bin

# Functie pentru a aplica prelungirea muchiilor prin histereza
def prelungire_muchii_histereza(img, lowThreshold, highThreshold):
    # Functie interna pentru non-maximum suppression
    def non_max_suppression(img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        for i in range(1, M-1):
            for j in range(1, N-1):
                if D[i, j] == 0:
                    Z[i, j] = img[i, j]
                else:
                    if img[i, j] > img[i + D[i, j][0], j + D[i, j][1]] and img[i, j] > img[i - D[i, j][0], j - D[i, j][1]]:
                        Z[i, j] = img[i, j]
        return Z

    # Functie interna pentru histereza
    def hysteresis(img, low, high):
        M, N = img.shape
        strong = 255
        weak = 100
        img[img > high] = strong
        img[np.logical_and(low <= img, img <= high)] = weak
        img[img < low] = 0

        for i in range(1, M-1):
            for j in range(1, N-1):
                if img[i, j] == weak:
                    if np.any(img[i-1:i+2, j-1:j+2] == strong):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
        return img

    # Convertește imaginea in nivele de gri si normalizeaza valorile pixelilor
    img_gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    img_gri = img_gri / 255.0

    # Aplica filtrul Sobel pentru a calcula gradientul pe axele X si Y
    gx = convolve(img_gri, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    gy = convolve(img_gri, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    # Calculeaza magnitudinea gradientului si normalizeaza valorile pixelilor
    img_grad = np.sqrt(gx**2 + gy**2)
    img_grad = (img_grad / np.max(img_grad)) * 255
    img_grad = img_grad.astype(np.uint8)
    # Calculeaza unghiul gradientului si normalizeaza valorile unghiurilor
    img_angle = np.arctan2(gy, gx) * 180 / np.pi
    img_angle = (img_angle + 180) % 180

    # Determina directia gradientului in functie de unghiul gradientului
    D = np.zeros(img_angle.shape, dtype=object)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if (0 <= img_angle[i, j] < 22.5) or (157.5 <= img_angle[i, j] < 180):
                D[i, j] = (0, 1)
            elif 22.5 <= img_angle[i, j] < 67.5:
                D[i, j] = (-1, 1)
            elif 67.5 <= img_angle[i, j] < 112.5:
                D[i, j] = (-1, 0)
            else:
                D[i, j] = (-1, -1)

    # Aplica non-maximum suppression pentru a subtia muchiile
    img_nms = non_max_suppression(img_grad, D)
    # Aplica histereza pentru a elimina muchiile slabe si a prelungi muchiile puternice
    img_canny = hysteresis(img_nms, lowThreshold, highThreshold)
    img_canny = img_canny.astype(np.uint8)

    return img_canny

# Incarca imaginea
imagine = cv2.imread('D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg')

# Aplica binarizarea adaptiva
blockSize = 15
C = 8
imagine_bin_adaptiva = binarizare_adaptiva(imagine, blockSize, C)

# Aplica prelungirea muchiilor prin histereza
lowThreshold = 50
highThreshold = 150
imagine_muchii_histereza = prelungire_muchii_histereza(imagine, lowThreshold, highThreshold)

cv2.imshow('Imagine originala', imagine)
cv2.imshow('Binarizare adaptiva', imagine_bin_adaptiva)
cv2.imshow('Prelungire muchii prin histereza', imagine_muchii_histereza)

cv2.waitKey(0)
cv2.destroyAllWindows()
