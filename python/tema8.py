import cv2
import numpy as np

def binarizare_automata_globala(img, error=0.01, max_iter=50):
    # Calculează histograma imaginii
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Găsește intensitățile maxime și minime
    i_min = np.argmax(hist)
    i_max = np.argmin(hist)
    
    # Inițializează pragul t
    t = (i_min + i_max) // 2
    
    # Iterații pentru actualizarea pragului t
    for i in range(max_iter):
        # Segmentează imaginea pe baza pragului t
        binary_img = np.zeros_like(img)
        binary_img[img >= t] = 255
        
        # Calculează valorile medii ale segmentelor
        mean1 = cv2.mean(img, binary_img)[0]
        mean2 = cv2.mean(img, cv2.bitwise_not(binary_img))[0]
        
        # Actualizează pragul
        new_t = (mean1 + mean2) // 2
        
        # Verifică condiția de oprire
        if abs(new_t - t) <= error:
            break
        
        t = new_t
    
    # Binarizează imaginea folosind pragul calculat
    binary_img = np.zeros_like(img)
    binary_img[img >= t] = 255
    
    return binary_img

def negativare_imagine(imagine):
    """Returneaza imaginea negativata."""
    imagine_negativata = 255 - imagine
    return imagine_negativata

def modifica_contrast(image, min_out=0, max_out=0):
    # determine minimum and maximum values in the image
    min_in = np.min(image)
    max_in = np.max(image)

    # calculate contrast scaling factor
    scale_factor = (max_out - min_out) / (max_in - min_in)

    # apply contrast scaling to the image
    modified_image = (image - min_in) * scale_factor + min_out

    return modified_image

def corectie_gamma(img, gamma=1.0):
    # Convertim imaginea în format de intensitate a griului
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalizăm intensitatea pixelilor la intervalul [0, 1]
    normalized_img = gray_img / 255.0

    # Aplicăm corecția gamma
    gamma_corrected_img = np.power(normalized_img, gamma)

    # Întoarcem imaginea la intervalul [0, 255] și convertim înapoi la format BGR
    gamma_corrected_img = (gamma_corrected_img * 255).astype(np.uint8)
    gamma_corrected_img = cv2.cvtColor(gamma_corrected_img, cv2.COLOR_GRAY2BGR)

    return gamma_corrected_img

def modifica_luminozitate(imagine, offset):
    """Returneaza imaginea negativata."""
    imagine_negativata = offset + imagine.astype(np.int32)
    imagine_negativata = np.clip(imagine_negativata, 0, 255).astype(np.uint8)
    return imagine_negativata


def egalizarea_histogramei(img):
    # Calculează histograma imaginii de intrare
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])

    # Calculează FDP-ul imaginii de intrare
    fdp = hist / np.sum(hist)

    # Calculează CDF-ul imaginii de intrare
    cdf = np.cumsum(fdp)

    # Calculează funcția de transformare
    tab = np.zeros(256)
    for i in range(256):
        tab[i] = int(255 * cdf[i] + 0.5)

    # Aplică funcția de transformare pe fiecare pixel al imaginii de intrare
    img_eq = tab[img]

    # Convertște imaginea egalizată înapoi în tipul de date uint8
    img_eq = np.uint8(img_eq)

    return img_eq


img_bw = cv2.imread('D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg', 0)
img = cv2.imread('D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.jpg')

binary_img = binarizare_automata_globala(img_bw)

negative_img = negativare_imagine(img)

contrast_img = modifica_contrast(img, 0, 4)

gamma_img = corectie_gamma(img)

luminozitate_img = modifica_luminozitate(img, 100)

egalizareHist_img = egalizarea_histogramei(img_bw)

# Afișează imaginea binarizată
cv2.imshow('Binarizare automata globala', binary_img)
cv2.imshow('Negativare', negative_img)
cv2.imshow('Modificare contrast', contrast_img)
cv2.imshow('Gamma', gamma_img)
cv2.imshow('Luminozitate', luminozitate_img)
cv2.imshow('Egalizare histograma', egalizareHist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()