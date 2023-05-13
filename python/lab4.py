import cv2 
import numpy as np
import random

img = cv2.imread("D:/wamp64/www/Prelucrarea Imaginilor/python/text_negru_pe_alb.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('Imaginea initiala', img)
label = 0
labels = np.zeros((img.shape[0], img.shape[1]))

def get_neighbors(i,j):
    i_min = max(i - 1, 0)
    i_max = min(i + 1 + 1, img.shape[0])
    j_min = max(j - 1, 0)
    j_max = min(j + 1 + 1, img.shape[1])

    neighbors = list((x,y) for x in range(i_min,i_max) for y in range(j_min,j_max) if x!=i or y !=j)
    return neighbors


for i in range(img.shape[0]):
    label += 1
    for j in range(img.shape[1]):
        if img[i,j] == 0 and labels[i,j] == 0:
            labels[i][j] = label
            queue = [(i, j)]

            while queue:
                q = queue.pop(0)
                if img[q[0]][q[1]] == 0:
                    neighbors = get_neighbors(q[0], q[1])
                    for neighbor in neighbors:
                        if labels[neighbor[0]][neighbor[1]] == 0:
                            labels[neighbor[0]][neighbor[1]] = label
                            if neighbor not in queue:
                                queue.append(neighbor)

colors = []
for i in range(label):
    colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))


color_labels = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if labels[i][j] != 0:
            color_labels[i][j] = colors[int(labels[i][j])-1]

cv2.imshow('Traversarea in latime', color_labels)



def n_P_neighbors(i, j):
    return [(i, j-1), (i-1, j-1), (i-1, j), (i-1, j+1)]

# Doua treceri de clase de echivalenta
label = 0
labels = np.zeros((img.shape[0], img.shape[1]))
edges =[[] for i in range (img.shape[0]*img.shape[1])]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] == 0 and labels[i][j] == 0:
            L = []
            neighbors = n_P_neighbors(i, j)
            for neighbor in neighbors:
                if labels[neighbor[0]][neighbor[1]] > 0:
                    L.append(labels[neighbor[0]][neighbor[1]])
                if len(L) == 0:
                    label += 1
                    labels[i][j] = label
                else:
                    x = min(L).astype(int)
                    labels[i][j] = x
                    for y in L:
                        if y != x:
                            edges[x].append(y.astype(int))
                            edges[y.astype(int)].append(x)

newlabel = 0
newlabels = np.zeros(img.shape[0]*img.shape[1], dtype=np.int32)
for i in range(len(edges)):
    if newlabels[i] == 0:
        newlabel += 1
        queue = [i]
        newlabels[i] = newlabel
        while queue:
            n = queue.pop()
            for e in edges[n]:
                if newlabels[e] == 0:
                    newlabels[e] = newlabel
                    queue.append(e)


for i  in range(img.shape[0]):
    for j in range(img.shape[1]):
        if labels[i][j] > 0:
            labels[i][j] = newlabels[labels[i][j].astype(int)].astype(int)

labels.astype(int)

colors = []
for i in range(newlabel):
    colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
color_labels = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if labels[i][j] != 0:
            color_labels[i][j] = colors[int(labels[i][j])-1]

cv2.imshow('Doua treceri cu clase de echivalenta', color_labels)

cv2.waitKey(0)
cv2.destroyAllWindows()
