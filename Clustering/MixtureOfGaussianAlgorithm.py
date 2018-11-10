from re import split
import numpy as np
import math
import matplotlib.pyplot as plt


class MixtureOfGaussianAlgorithm(object):
    def __init__(self):
        filename = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\dataset\Watermelon\Watermelon4.txt"
        self.data = self.load_data(filename)

        self.iter = 50
        self.res = []

    # Read sample set from Watermelon4.txt
    def load_data(self, filename):
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
    def guass_prob(self, x, u, sigma):
        n = 2
        e_power = -0.5 * (x - u) * sigma.I * np.matrix(x - u).T
        denominator = pow(2 * math.pi, n / 2) * pow(np.linalg.det(sigma), 0.5)
        p = pow(math.e, e_power[0, 0]) / denominator
        return p

    # Update parameters of the gaussian distribution
    def em(self):
        m = len(self.data)
        k = 3
        # Initiate the parameters of Mixture-of-Gaussian model
        alpha = [1 / 3, 1 / 3, 1 / 3]
        u = [self.data[5], self.data[21], self.data[26]]
        sigma = [np.matrix('0.1 0.0; 0.0 0.1') for x in range(3)]  # [matrix]
        gamma = np.zeros((m, 3))  # array
        # EM algorithm
        for xc in range(self.iter):
            for j in range(m):
                sum_alpha = 0
                for kc1 in range(k):
                    gamma[j][kc1] = alpha[kc1] * self.guass_prob(self.data[j], u[kc1], sigma[kc1])
                    sum_alpha += gamma[j, kc1]
                for kc2 in range(k):
                    gamma[j][kc2] /= sum_alpha
            sum_gamma = np.sum(gamma, axis=0)
            for kc3 in range(k):
                # Calculate new mean vector
                sum_u = np.array([0, 0], dtype=float)
                for jc1 in range(m):
                    sum_u += gamma[jc1][kc3] * self.data[jc1]
                u[kc3] = sum_u / sum_gamma[kc3]
                # Calculate new covariance matrix (sigma)
                sum_sigma = np.matrix('0.0 0.0; 0.0 0.0')
                for jc2 in range(m):
                    sum_sigma += gamma[jc2][kc3] * np.matrix(self.data[jc2] - u[kc3]).T * np.matrix(self.data[jc2] - u[kc3])
                sigma[kc3] = sum_sigma / sum_gamma[kc3]
                # Calculate new mixing factor (alpha)
                alpha[kc3] = sum_gamma[kc3] / m
        return gamma

    # Put the sample into corresponding cluster
    def cluster(self):
        m = len(self.data)
        self.res = np.zeros((m, 2))
        gamma = self.em()
        for i in range(m):
            self.res[i, :] = np.argmax(gamma[i, :]), np.amax(gamma[i, :])

    # Visualization
    def visualization(self):
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in range(len(self.data)):
            if self.res[i, 0] == 0:
                x0.append(self.data[i][0])
                y0.append(self.data[i][1])
            if self.res[i, 0] == 1:
                x1.append(self.data[i][0])
                y1.append(self.data[i][1])
            if self.res[i, 0] == 2:
                x2.append(self.data[i][0])
                y2.append(self.data[i][1])
        plt.scatter(x0, y0, c='r', alpha=0.5)
        plt.scatter(x1, y1, c='b', alpha=0.5)
        plt.scatter(x2, y2, c='g', alpha=0.5)

        plt.xlim(0.1, 0.9)
        plt.ylim(0, 0.8)
        plt.show()

    def execute(self):
        self.cluster()
        self.visualization()
