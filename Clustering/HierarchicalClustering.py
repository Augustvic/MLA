from re import split
import numpy as np
import math
import matplotlib.pyplot as plt


class HierarchicalClustering(object):
    def __init__(self):
        filename = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\dataset\Watermelon\Watermelon4.txt"
        self.dataset = self.load_data(filename)

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
            D.append(e)
        return D

    # Calculate the distance between two sample
    def dist(self, sample1, sample2):
        distance = math.pow(math.pow(float(sample1[0]) - float(sample2[0]), 2) +
                        math.pow(float(sample1[1]) - float(sample2[1]), 2), 0.5)
        return distance

    # Calculate the distance between two cluster (max)
    def distance_max(self, Ci, Cj):
        dist_between_cluster = []
        for ci in Ci:
            for cj in Cj:
                dist_between_cluster.append(self.dist(ci, cj))
        return max(dist_between_cluster)

    # Hierarchical Clustering(AGNES) algorithm
    def hierarchical(self, data, k):
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
                M[i, j] = self.distance_max(C[i], C[j])
                M[j, i] = M[i, j]
        q = m
        while q > k:
            min_dist = M[0, 1]
            min_i = 0
            min_j = 1
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if i != j:
                        if M[i, j] < min_dist:
                            min_dist = M[i, j]
                            min_i = i
                            min_j = j
            for v in C[min_j]:
                C[min_i].append(v)
            for j in range(min_j + 1, q):
                C[j-1] = C.pop(j)
            M = np.delete(M, min_j, axis=0)
            M = np.delete(M, min_j, axis=1)
            for j in range(0, q-1):
                M[min_i, j] = self.distance_max(C[min_i], C[j])
                M[j, min_i] = M[min_i, j]
            q -= 1
        return C

    def visualization(self, res):
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

    def execute(self):
        k = 5
        res = self.hierarchical(self.dataset, k)
        self.visualization(res)
