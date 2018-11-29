from re import split
import numpy as np
import matplotlib.pyplot as plt


class KMeansAlgorithm(object):
    def __init__(self):
        # Read sample set D from Watermelon4.txt
        filename = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Watermelon\Watermelon4.txt"
        self.data = self.load_data(filename)

        self.nums = 5
        self.k = 3

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

    def update(self, P):
        # Update clustering for "nums" times
        for no in range(self.nums):
            # Divide the sample set into k clusters, and store them into 'res'
            res = []
            for i in range(self.k):
                tmp = []
                res.append(tmp)

            print("--")
            print(res)
            print(P)

            # Put the sample into corresponding cluster according to the
            # distance between the sample and mean vector
            for i in range(len(self.data)):
                min_index = 0
                dist0 = self.data[i] - P[0]
                min_res = (dist0[0] ** 2 + dist0[1] ** 2) ** 0.5
                for j in range(self.k):
                    dist = self.data[i] - P[j]
                    distance = (dist[0] ** 2 + dist[1] ** 2) ** 0.5
                    print(distance, "-", j)
                    if distance < min_res:
                        min_index = j
                        min_res = distance
                print(min_res, "+", min_index)
                res[min_index].append(self.data[i])

            print("--")
            print(res)
            print(P)

            # Calculate the new mean vector, to decide whether to replace
            for l in range(self.k):
                sum_num = np.array([0, 0], dtype=float)
                for sum_ele in res[l]:
                    sum_num += sum_ele
                u = (1 / len(res[l])) * sum_num
                if not np.array_equal(P[l], u):
                    P[l] = u

            print("--")
            print(res)
            print(P)

            self.visualization(res)

    def visualization(self, res):
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

    def execute(self):
        # Initiate mean vector
        P = []
        e1 = np.array([0.403, 0.237], dtype=float)
        P.append(e1)
        e2 = np.array([0.343, 0.099], dtype=float)
        P.append(e2)
        e3 = np.array([0.532, 0.472], dtype=float)
        P.append(e3)
        self.update(P)


"""
P = []
k = 3
for i in range(k):
    j = random.randint(0, len(D)-1)
    e = np.array(copy.deepcopy(D[j]), dtype=float)
    P.append(e)
"""