import cv2
import numpy as np

def urmarire_contur(imagine_binara):
    # Verifică versiunea OpenCV și apelează funcția findContours corespunzător
    (major, minor, _) = cv2.__version__.split(".")
    if int(major) < 4:
        _, contururi, _ = cv2.findContours(imagine_binara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contururi, _ = cv2.findContours(imagine_binara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Creează o imagine nouă pentru a desena contururile
    imagine_contururi = np.zeros_like(imagine_binara)

    # Desenează contururile pe imaginea nouă
    cv2.drawContours(imagine_contururi, contururi, -1, 255, 1)

    return imagine_contururi

def gaseste_punct_start(contur):
    for punct in contur:
        if punct[0][1] == min(contur, key=lambda x: x[0][1])[0][1]:
            return tuple(punct[0])

def calculeaza_cod_lant(prev_direction, current_point, next_point):
    deltas = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    x_dif, y_dif = next_point[0] - current_point[0], next_point[1] - current_point[1]

    for index, delta in enumerate(deltas):
        if delta == (x_dif, y_dif):
            return (prev_direction + index) % 8

def extrage_cod_lant(imagine_binara):
    # Verifică versiunea OpenCV și apelează funcția findContours corespunzător
    (major, minor, _) = cv2.__version__.split(".")
    if int(major) < 4:
        _, contururi, _ = cv2.findContours(imagine_binara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contururi, _ = cv2.findContours(imagine_binara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    coduri_lant = []
    for contur in contururi:
        punct_start = gaseste_punct_start(contur)
        punct_curent = punct_start
        index_curent = np.where((contur == punct_start).all(axis=2))[0][0]
        cod_lant = []

        while True:
            punct_urmator_index = (index_curent + 1) % len(contur)
            punct_urmator = tuple(contur[punct_urmator_index][0])

            if len(cod_lant) == 0:
                prev_direction = 0
            else:
                prev_direction = cod_lant[-1]

            cod = calculeaza_cod_lant(prev_direction, punct_curent, punct_urmator)
            cod_lant.append(cod)

            index_curent = punct_urmator_index
            punct_curent = punct_urmator

            if punct_curent == punct_start:
                break

        coduri_lant.append(cod_lant)

    return coduri_lant

# Încarcă imaginea normală
imagine_sursa = cv2.imread("D:\wamp64\www\Prelucrarea Imaginilor\python\penguins.png")

# Convertește imaginea în alb-negru
imagine_gri = cv2.cvtColor(imagine_sursa, cv2.COLOR_BGR2GRAY)

# Aplică pragul pentru a obține o imagine binară
_, imagine_binara = cv2.threshold(imagine_gri, 127, 255, cv2.THRESH_BINARY)

# Aplică algoritmul de urmărire a conturului
imagine_contururi = urmarire_contur(imagine_binara)

# Afișează imaginile
cv2.imshow("Imaginea initiala", imagine_sursa)
cv2.imshow("Imaginea binara", imagine_binara)
cv2.imshow("Contururile", imagine_contururi)

# Extrage codurile în lanț
coduri_lant = extrage_cod_lant(imagine_binara)

# Afișează codurile în lanț pentru fiecare obiect
for index, cod_lant in enumerate(coduri_lant):
    print(f"Codul în lanț pentru obiectul {index + 1} este {cod_lant}")

# Așteaptă apăsarea unei taste și închide toate ferestrele
cv2.waitKey(0)
cv2.destroyAllWindows()
