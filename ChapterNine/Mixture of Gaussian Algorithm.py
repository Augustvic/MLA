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
        e = np.array(e, dtype=float)
        D.append(e)
    return D


# Gaussian distribution probability
def guassProb(x, u, sigma):
    n = 2
    epower = -0.5 * (x - u) * sigma.I * np.matrix(x - u).T
    denominator = pow(2 * math.pi, n/2) * pow(np.linalg.det(sigma), 0.5)
    p = pow(math.e, epower[0, 0]) / denominator
    return p


# Update parameters of the gaussian distribution
def EM(dataset, iter):
    m = len(dataset)
    k = 3
    # Initiate the parameters of Mixture-of-Gaussian model
    alpha = [1/3, 1/3, 1/3]
    u = [dataset[5], dataset[21], dataset[26]]
    sigma = [np.matrix('0.1 0.0; 0.0 0.1') for x in range(3)]     # [matrix]
    gamma = np.zeros((m, 3))     # array
    # EM algorithm
    for xc in range(iter):
        for j in range(m):
            sumAlpha = 0
            for kc1 in range(k):
                gamma[j][kc1] = alpha[kc1] * guassProb(dataset[j], u[kc1], sigma[kc1])
                sumAlpha += gamma[j, kc1]
            for kc2 in range(k):
                gamma[j][kc2] /= sumAlpha
        sumGamma = np.sum(gamma, axis=0)
        for kc3 in range(k):
            # Calculate new mean vector
            sumu = np.array([0, 0], dtype=float)
            for jc1 in range(m):
                sumu += gamma[jc1][kc3] * dataset[jc1]
            u[kc3] = sumu / sumGamma[kc3]
            # Calculate new covariance matrix (sigma)
            sumsigma = np.matrix('0.0 0.0; 0.0 0.0')
            for jc2 in range(m):
                sumsigma += gamma[jc2][kc3] * np.matrix(dataset[jc2] - u[kc3]).T * np.matrix(dataset[jc2] - u[kc3])
            sigma[kc3] = sumsigma / sumGamma[kc3]
            # Calculate new mixing factor (alpha)
            alpha[kc3] = sumGamma[kc3] / m
    return gamma


# Put the sample into corresponding cluster
def cluster(dataset):
    m = len(dataset)
    res = np.zeros((m, 2))
    gamma = EM(dataset, 50)
    for i in range(m):
        res[i, :] = np.argmax(gamma[i, :]), np.amax(gamma[i, :])
    return res


filename = r"C:\Users\August\August's Documents\MachineLearningWatermelon\Watermelon4.txt"
dataset = loadData(filename)
res = cluster(dataset)

# Visualization
x0 = []
y0 = []
x1 = []
y1 = []
x2 = []
y2 = []
for i in range(len(dataset)):
    if res[i, 0] == 0:
        x0.append(dataset[i][0])
        y0.append(dataset[i][1])
    if res[i, 0] == 1:
        x1.append(dataset[i][0])
        y1.append(dataset[i][1])
    if res[i, 0] == 2:
        x2.append(dataset[i][0])
        y2.append(dataset[i][1])
plt.scatter(x0, y0, c='r', alpha=0.5)
plt.scatter(x1, y1, c='b', alpha=0.5)
plt.scatter(x2, y2, c='g', alpha=0.5)

plt.xlim(0.1, 0.9)
plt.ylim(0, 0.8)
plt.show()
