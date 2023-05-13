import cv2
import numpy as np

def extrage_contur(image):
    # Convertește imaginea în tonuri de gri
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplică filtrarea Gaussiană pentru a reduce zgomotul
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detectează marginile folosind algoritmul Canny
    edges = cv2.Canny(blurred_image, 100, 200)

    # Găsește contururile din imagine
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenează contururile pe imaginea originală
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    return contour_image

# Încarcă imaginea
image_path = "D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg"
color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Aplică extragerea conturului
contour_image = extrage_contur(color_image)

# Afișează imaginea originală și imaginea cu contururi extrase
cv2.imshow("Imaginea originala", color_image)
cv2.imshow("Imaginea cu contururi extrase", contour_image)

def umple_regiune(image, seed_point, new_value, tolerance=1):
    original_value = image[seed_point[1], seed_point[0]]
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(image, mask, seed_point, new_value, (tolerance, tolerance, tolerance),
                  (tolerance, tolerance, tolerance), cv2.FLOODFILL_MASK_ONLY)

    filled_region = cv2.bitwise_and(image, image, mask=mask[1:-1, 1:-1])
    return filled_region

# Încarcă imaginea
image_path = "D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg"
color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Definește punctul de pornire și culoarea pentru umplerea regiunii
seed_point = (30, 30)  # (x, y)
new_value = (0, 255, 0)  # Verde

# Aplică umplerea regiunii
filled_region = umple_regiune(color_image, seed_point, new_value)

# Afișează imaginea originală și imaginea cu regiunea umplută
cv2.imshow("Imaginea originala", color_image)
cv2.imshow("Regiunea umpluta", filled_region)
cv2.waitKey(0)
cv2.destroyAllWindows()