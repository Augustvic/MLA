from re import split
import numpy as np
import random
import matplotlib.pyplot as plt


class LearningVectorQuantization(object):
    def __init__(self):
        # Read sample set D from Watermelon4.txt
        filename = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Watermelon\Watermelon4.txt"
        self.data = self.load_data(filename)

        self.lRate = 0.1  # Learning Rate
        self.k = 5
        self.res = []

        # Initiate prototype vector
        self.P = []
        e1 = np.array([0.556, 0.215, 1], dtype=float)
        self.P.append(e1)
        e2 = np.array([0.343, 0.099, 0], dtype=float)
        self.P.append(e2)
        e3 = np.array([0.359, 0.188, 0], dtype=float)
        self.P.append(e3)
        e4 = np.array([0.483, 0.312, 1], dtype=float)
        self.P.append(e3)
        e5 = np.array([0.725, 0.445, 1], dtype=float)
        self.P.append(e3)

    def load_data(self, filename):
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
        return D

    # Update prototype vector
    def update(self):
        for i in range(50):
            j = random.randint(0, len(self.data) - 1)
            min_index = 0
            dist0 = self.data[j] - self.P[0]
            min_res = (dist0[0] ** 2 + dist0[1] ** 2) ** 0.5
            for m in range(self.k):
                dist = self.data[j] - self.P[m]
                distance = (dist[0] ** 2 + dist[1] ** 2) ** 0.5
                if distance < min_res:
                    min_res = distance
                    min_index = m
            tag_d = self.data[j][2]
            tag_p = self.P[min_index][2]
            if tag_d == tag_p:
                p3 = self.P[min_index][2]
                self.P[min_index] = self.P[min_index] + self.lRate * (self.data[j] - self.P[min_index])
                self.P[min_index][2] = p3
            else:
                p3 = self.P[min_index][2]
                self.P[min_index] = self.P[min_index] - self.lRate * (self.data[j] - self.P[min_index])
                self.P[min_index][2] = p3

    # Put the sample into corresponding cluster according to the
    # distance between the sample and prototype vector
    def quantization(self):
        self.res = []
        for i in range(self.k):
            tmp = []
            self.res.append(tmp)
        for i in range(len(self.data)):
            min_index = 0
            dist0 = self.data[i] - self.P[0]
            min_dis = (dist0[0] ** 2 + dist0[1] ** 2) ** 0.5
            for j in range(len(self.P)):
                dist = self.data[i] - self.P[j]
                distance = (dist[0] ** 2 + dist[1] ** 2) ** 0.5
                if distance < min_dis:
                    min_dis = distance
                    min_index = j
            self.res[min_index].append(self.data[i])

    def visualization(self):
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
        for point in self.res[0]:
            x1.append(point[0])
            y1.append(point[1])
        for point in self.res[1]:
            x2.append(point[0])
            y2.append(point[1])
        for point in self.res[2]:
            x3.append(point[0])
            y3.append(point[1])
        for point in self.res[3]:
            x4.append(point[0])
            y4.append(point[1])
        for point in self.res[4]:
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
        self.update()
        self.quantization()
        self.visualization()
