from re import split
import numpy as np
import random
import matplotlib.pyplot as plt

# Read sample set D from Watermelon4.txt
filename = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\dataset\Watermelon\Watermelon4.txt"
delim = ' '
with open(filename) as f:
    data = f.readlines()
D = []
for no, line in enumerate(data):
    e = []
    items = split(delim, line.strip())
    e.append(items[1])
    e.append(items[2])
    # Set category tag for every sample
    if 7 < no < 21:
        e.append('0')
    else:
        e.append('1')
    e = np.array(e, dtype=float)
    D.append(e)

# Initiate prototype vector
k = 5
P = []
e1 = np.array([0.556, 0.215, 1], dtype=float)
P.append(e1)
e2 = np.array([0.343, 0.099, 0], dtype=float)
P.append(e2)
e3 = np.array([0.359, 0.188, 0], dtype=float)
P.append(e3)
e4 = np.array([0.483, 0.312, 1], dtype=float)
P.append(e3)
e5 = np.array([0.725, 0.445, 1], dtype=float)
P.append(e3)

lRate = 0.1     # Learning Rate
# Update prototype vector
for i in range(400):
    j = random.randint(0, len(D) - 1)
    minIndex = 0
    dist0 = D[j] - P[0]
    minRes = (dist0[0] ** 2 + dist0[1] ** 2) ** 0.5
    for m in range(k):
        dist = D[j] - P[m]
        distance = (dist[0] ** 2 + dist[1] ** 2) ** 0.5
        if distance < minRes:
            minRes = distance
            minIndex = m
    tagD = D[j][2]
    tagP = P[minIndex][2]
    if tagD == tagP:
        p3 = P[minIndex][2]
        P[minIndex] = P[minIndex] + lRate * (D[j] - P[minIndex])
        P[minIndex][2] = p3
    else:
        p3 = P[minIndex][2]
        P[minIndex] = P[minIndex] - lRate * (D[j] - P[minIndex])
        P[minIndex][2] = p3

# Put the sample into corresponding cluster according to the
# distance between the sample and prototype vector
res = []
for i in range(k):
    tmp = []
    res.append(tmp)
for i in range(len(D)):
    minIndex = 0
    dist0 = D[i] - P[0]
    minDis = (dist0[0] ** 2 + dist0[1] ** 2) ** 0.5
    for j in range(len(P)):
        dist = D[i] - P[j]
        distance = (dist[0] ** 2 + dist[1] ** 2) ** 0.5
        if distance < minDis:
            minDis = distance
            minIndex = j
    res[minIndex].append(D[i])

# Visualization
x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []
x5 = []
y5 = []
for point in res[0]:
    x1.append(point[0])
    y1.append(point[1])
for point in res[1]:
    x2.append(point[0])
    y2.append(point[1])
for point in res[2]:
    x3.append(point[0])
    y3.append(point[1])
for point in res[3]:
    x4.append(point[0])
    y4.append(point[1])
for point in res[4]:
    x5.append(point[0])
    y5.append(point[1])

plt.scatter(x1, y1, c='r', alpha=0.5)
plt.scatter(x2, y2, c='b', alpha=0.5)
plt.scatter(x3, y3, c='g', alpha=0.5)
plt.scatter(x4, y4, c='y', alpha=0.5)
plt.scatter(x5, y5, c='m', alpha=0.5)

plt.xlim(0.1, 0.9)
plt.ylim(0, 0.8)
plt.show()
