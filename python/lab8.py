import cv2
import numpy as np

def binarizare_automatica_globala(image, initial_threshold=128, delta_t=1):
    threshold = initial_threshold
    new_threshold = 0
    
    while True:
        lower_region = image[image <= threshold]
        upper_region = image[image > threshold]
        
        if len(lower_region) == 0 or len(upper_region) == 0:
            break

        lower_region_mean = np.mean(lower_region)
        upper_region_mean = np.mean(upper_region)

        new_threshold = (lower_region_mean + upper_region_mean) / 2

        if abs(new_threshold - threshold) < delta_t:
            break

        threshold = new_threshold

    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# Încarcă imaginea color
image_path = "D:\wamp64\www\Prelucrarea Imaginilor\python\imagine.jpeg"
color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convertește imaginea în tonuri de gri
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Aplică binarizarea automată globală
binary_image = binarizare_automatica_globala(gray_image)

# Afișează imaginea originală, imaginea în tonuri de gri și imaginea binarizată
cv2.imshow("Imaginea originala", color_image)
cv2.imshow("Imaginea in tonuri de gri", gray_image)
cv2.imshow("Imaginea binarizata", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
