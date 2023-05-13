import cv2
import numpy as np
import time

def getGaussianKernel(kernel_size, sigma):
    # Initializarea unui kernel gol
    kernel = np.zeros((kernel_size, 1))
    center = kernel_size // 2
    # Popularea kernelului cu valori Gaussiene
    for i in range(kernel_size):
        kernel[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
    # Normalizarea kernelului
    return kernel / np.sum(kernel)

def restaurare_imagine_gauss(imagine):
    # Aplicarea filtrului Gaussian pe imagine
    kernel = getGaussianKernel(5,1)
    imagine_filtrata = cv2.filter2D(imagine, -1, kernel)
    return imagine_filtrata

# Start timp pentru a calcula durata procesului
start_time = time.time()

img = cv2.imread("D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg")
img_restaurata_gauss = restaurare_imagine_gauss(img)

print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow("Imaginea originala", img)
cv2.imshow("Imaginea filtrata", img_restaurata_gauss)

def outer(kernel):
    # Realizeaza produsul exterior al unui kernel
    n = kernel.shape[0]
    output = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            output[i, j] = kernel[i] * kernel[j]
    return output

def restaurare_imagine_bidimensional(imagine):
    # Aplica filtrul Gaussian bidimensional pe imagine
    kernel = getGaussianKernel(5, 0.1)
    kernel_2d = outer(kernel)

    imagine_filtrata = cv2.filter2D(imagine, -1, kernel_2d)

    return imagine_filtrata

# Start timp pentru a calcula durata procesului
start_time = time.time()
img_restaurata_bidimensional = restaurare_imagine_bidimensional(img)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow("Imaginea restaurata", img_restaurata_bidimensional)
cv2.waitKey(0)
cv2.destroyAllWindows()