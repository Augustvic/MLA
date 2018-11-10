from re import split
import numpy as np
import math
import copy
from random import choice
from queue import Queue
import matplotlib.pyplot as plt


class DensityBasedClustering(object):
    def __init__(self):
        filename = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\dataset\Watermelon\Watermelon4.txt"
        self.data = self.load_data(filename)
        self.para = [0.11, 5]

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

    # Count the number of samples in the neighborhood of x
    def count_neighbors(self, data, x, epsilon):
        m = len(data)
        count = 0
        for i in range(m):
            if self.dist(x, data[i]) <= epsilon:
                count += 1
        return count

    # The neighborhoods of x
    def neighborhood(self, data, x, epsilon):
        m = len(data)
        neigh = []
        for i in range(m):
            if self.dist(x, data[i]) <= epsilon:
                neigh.append(data[i])
        return neigh

    # Density-based Clustering algorithm
    def density_based(self, data, para):
        m = len(data)
        omega = []
        epsilon = para[0]
        min_pts = para[1]
        for j in range(m):
            count = self.count_neighbors(data, data[j], epsilon)
            if count >= min_pts:
                omega.append(data[j])
        k = 0     # the number of cluster
        gamma = copy.deepcopy(data)     # the set of unvisited sample
        cluster = {}      # the result of cluster division

        while omega:
            gamma_old = copy.deepcopy(gamma)
            o = choice(omega)
            Q = Queue()
            Q.put(o)
            gamma.remove(o)
            while not Q.empty():
                q = Q.get()
                neigh = self.neighborhood(data, q, epsilon)
                if len(neigh) >= min_pts:
                    intersection = [v for v in neigh if v in gamma]
                    for inters in intersection:
                        Q.put(inters)
                        gamma.remove(inters)
            cluster[k] = [v for v in gamma_old if v not in gamma]
            omega = [v for v in omega if v not in cluster[k]]

            # Visualization
            x0 = []
            y0 = []
            for ii in cluster[k]:
                x0.append(ii[0])
                y0.append(ii[1])
            plt.scatter(x0, y0, c='b', alpha=0.5)
            plt.xlim(0.1, 0.9)
            plt.ylim(0, 0.8)
            plt.show()

            k += 1
        return cluster

    def execute(self):
        res = self.density_based(self.data, self.para)
        print(res)
