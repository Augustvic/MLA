from re import split
import numpy as np
import matplotlib.pyplot as plt


# Read sample set D from Watermelon4.txt
filename = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\dataset\Watermelon\Watermelon4.txt"
delim = ' '
with open(filename) as f:
    data = f.readlines()
D = []
for line in data:
    e = []
    items = split(delim, line.strip())
    e.append(items[1])
    e.append(items[2])
    e = np.array(e, dtype=float)
    D.append(e)

"""
P = []
k = 3
for i in range(k):
    j = random.randint(0, len(D)-1)
    e = np.array(copy.deepcopy(D[j]), dtype=float)
    P.append(e)
"""

# Initiate mean vector
k = 3
P = []
e1 = np.array([0.403, 0.237], dtype=float)
P.append(e1)
e2 = np.array([0.343, 0.099], dtype=float)
P.append(e2)
e3 = np.array([0.478, 0.437], dtype=float)
P.append(e3)


# Update clustering for 4 times
for no in range(5):
    # Divide the sample set into k clusters, and store them into 'res'
    res = []
    for i in range(k):
        tmp = []
        res.append(tmp)

    # Put the sample into corresponding cluster according to the
    # distance between the sample and mean vector
    for i in range(len(D)):
        minIndex = 0
        dist0 = D[i] - P[0]
        minRes = (dist0[0] ** 2 + dist0[1] ** 2) ** 0.5
        for j in range(k):
            dist = D[i] - P[j]
            distance = (dist[0] ** 2 + dist[1] ** 2) ** 0.5
            if distance < minRes:
                minIndex = j
                minRes = distance
        res[minIndex].append(D[i])

    # Calculate the new mean vector, to decide whether to replace
    for l in range(k):
        sumNum = np.array([0, 0], dtype=float)
        for sumEle in res[l]:
            sumNum += sumEle
        u = (1 / len(res[l])) * sumNum
        if not np.array_equal(P[l], u):
            P[l] = u

    # Visualization
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    for point in res[0]:
        x1.append(point[0])
        y1.append(point[1])
    for point in res[1]:
        x2.append(point[0])
        y2.append(point[1])
    for point in res[2]:
        x3.append(point[0])
        y3.append(point[1])

    plt.scatter(x1, y1, c='r', alpha=0.5)
    plt.scatter(x2, y2, c='b', alpha=0.5)
    plt.scatter(x3, y3, c='g', alpha=0.5)

    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)
    plt.show()
