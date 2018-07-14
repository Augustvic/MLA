from re import split
import numpy as np
import math
import matplotlib.pyplot as plt


# Read sample set from Watermelon4.txt
def loadData(filename):
    delim = ' '
    with open(filename) as f:
        data = f.readlines()
    D = []
    for line in data:
        e = []
        items = split(delim, line.strip())
        e.append(items[1])
        e.append(items[2])
        D.append(e)
    return D


# Calculate the distance between two sample
def dist(sample1, sample2):
    distance = math.pow(math.pow(float(sample1[0]) - float(sample2[0]), 2) +
                        math.pow(float(sample1[1]) - float(sample2[1]), 2), 0.5)
    return distance


# Calculate the distance between two cluster (max)
def distanceMax(Ci, Cj):
    distBetweenCluster = []
    for ci in Ci:
        for cj in Cj:
            distBetweenCluster.append(dist(ci, cj))
    return max(distBetweenCluster)


# Hierarchical Clustering(AGNES) algorithm
def hierarchical(data, k):
    m = len(data)
    C = {}
    # Initiate single-sample clusters
    for j in range(m):
        Cj = []
        Cj.append(data[j])
        C[j] = Cj
    # Initiate cluster distance matrix
    M = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            M[i, j] = distanceMax(C[i], C[j])
            M[j, i] = M[i, j]
    q = m
    while q > k :
        minDist = M[0, 1]
        minI = 0
        minJ = 1
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i != j:
                    if M[i, j] < minDist:
                        minDist = M[i, j]
                        minI = i
                        minJ = j
        for v in C[minJ]:
            C[minI].append(v)
        for j in range(minJ + 1, q):
            C[j-1] = C.pop(j)
        M = np.delete(M, minJ, axis=0)
        M = np.delete(M, minJ, axis=1)
        for j in range(0, q-1):
            M[minI, j] = distanceMax(C[minI], C[j])
            M[j, minI] = M[minI, j]
        q -= 1
    return C



filename = r"C:\Users\August\August's Documents\MachineLearningWatermelon\Watermelon4.txt"
dataset = loadData(filename)
res = hierarchical(dataset, 5)

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
