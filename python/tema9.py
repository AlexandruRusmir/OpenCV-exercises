import numpy as np
import cv2

def low_pass_filter(img):
    # definim kernelul de 3x3 cu coeficienti 1/9
    kernel = np.ones((3,3),np.float32)/9
    
    # aplicam filtrul de convolutie
    result = cv2.filter2D(img,-1,kernel)
    
    # scalam rezultatul prin impartirea cu suma coeficientilor
    result = result/np.sum(kernel)
    
    # returnam imaginea filtrata
    return result.astype(np.uint8)


def high_pass_filter(img):
    # Definim kernel-ul de convolutie
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    # Aplicam filtrul
    filtered_img = cv2.filter2D(img, -1, kernel)
    # Scaleaza valorile din imagine in intervalul [0, 255]
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return filtered_img


img_bw = cv2.imread('D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg', 0)

img1 = low_pass_filter(img_bw)
img2 = high_pass_filter(img_bw)

cv2.imshow('Low pass', img1)
cv2.imshow('High pass', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()