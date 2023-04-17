import cv2
import numpy as np
import random

img = cv2.imread('D:/wamp64/www/Prelucrarea Imaginilor/python/figuriGeometrice.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)
  
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Conturul figurilor', img)
contours = [np.array(c, dtype=np.float32) for c in contours]

perimeter = []
area = []
for i, contour in enumerate(contours):
    moments = cv2.moments(contour)

    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        a = cv2.contourArea(contour)
        area.append(a)

        p = cv2.arcLength(contour, True)
        perimeter.append(p)

        ellipse = cv2.fitEllipse(contour)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.circle(img, (cx, cy), 5, color, -1)
        cv2.putText(img, f'Object {i+1}', (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.ellipse(img, ellipse, color, 2)

for i in range(len(area)):
    print("Aria elementului " + str(i) + ': ' + str(area[i]))

for i in range(len(perimeter)):
    print("Perimetrul elementului " + str(i) + ': ' + str(perimeter[i]))

cv2.imshow('Rezultat', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
