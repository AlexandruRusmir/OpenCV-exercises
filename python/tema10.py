import numpy as np
import cv2

def filtru_low_pass(img, radius):
    # Calculează dimensiunea imaginii și centrul imaginii
    rows, cols = img.shape[:2]
    center_row, center_col = rows//2, cols//2
    # Realizează transformata de centrare a imaginii
    shifted = np.fft.fftshift(np.fft.fft2(img))

    # Calculează distanțele de la centrul imaginii pentru fiecare punct
    y, x = np.ogrid[:rows, :cols]
    distances = np.sqrt((x - center_col)**2 + (y - center_row)**2)

    # Creează un filtru ideal de tip low pass
    low_pass_filter = np.zeros((rows, cols), dtype=np.uint8)
    low_pass_filter[distances <= radius] = 1

    # Aplică filtrul prin înmulțirea element cu element a matricii filtrului cu matricea shiftată
    filtered = shifted * low_pass_filter

    # Realizează transformata inversă a imaginii filtrate
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real

    # Realizează transformata de de-centrare a imaginii
    #filtered_image = np.roll(filtered_image, center_col, axis=1)
    #filtered_image = np.roll(filtered_image, center_row, axis=0)

    return filtered_image.astype(np.uint8)

img = cv2.imread("D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg", cv2.IMREAD_GRAYSCALE)
filtered_image = filtru_low_pass(img, 60)

cv2.imshow("Imaginea originala", img)
cv2.imshow("Low pass", filtered_image)

def filtru_high_pass(img, radius):
    # Calculează dimensiunea imaginii și centrul imaginii
    rows, cols = img.shape[:2]
    center_row, center_col = rows//2, cols//2
    # Realizează transformata de centrare a imaginii
    shifted = np.fft.fftshift(np.fft.fft2(img))

    # Calculează distanțele de la centrul imaginii pentru fiecare punct
    y, x = np.ogrid[:rows, :cols]
    distances = np.sqrt((x - center_col)*2 + (y - center_row)*2)

    # Creează un filtru ideal de tip high pass
    high_pass_filter = np.zeros((rows, cols), dtype=np.uint8)
    high_pass_filter[distances >= radius] = 1

    # Aplică filtrul prin înmulțirea element cu element a matricii filtrului cu matricea shiftată
    filtered = shifted * high_pass_filter

    # Realizează transformata inversă a imaginii filtrate
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real

    # Realizează transformata de de-centrare a imaginii
    # filtered_image = np.roll(filtered_image, center_col, axis=1)
    # filtered_image = np.roll(filtered_image, center_row, axis=0)

    return filtered_image.astype(np.uint8)

img = cv2.imread("D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg", cv2.IMREAD_GRAYSCALE)
filtered_image = filtru_high_pass(img, 2)

cv2.imshow("High pass", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()