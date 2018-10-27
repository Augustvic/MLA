# k-Nearest Neighbor Algorithm
# Created by August
# 2018/10/27
# Source:1.李航. 统计学习方法[M]. 清华大学出版社, 2012.
#        2.https://www.jianshu.com/p/be23b3870d2e


from re import split
import numpy as np
import math


class TreeNode(object):
    def __init__(self, val=[], left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class KNN(object):
    def __init__(self):
        allDataFilePath = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\Iris.txt"
        self.allData = self.load_data(allDataFilePath)
        # trainFilePath = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\train.txt"
        # self.train = self.load_data(trainFilePath)
        testFilePath = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\test.txt"
        self.test = self.load_data(testFilePath)
        self.train = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
        # self.train = np.array(self.train).astype(float)

        self.k = 2

    def load_data(self, filename):
        labels = {'Iris-setosa': '0', 'Iris-versicolor': '1', 'Iris-virginica': '2'}
        delim = ','
        with open(filename) as f:
            data = f.readlines()
        D = []
        for line in data:
            e = []
            items = split(delim, line.strip())
            for i in range(len(items)):
                if i == 4:
                    e.append(labels[items[i]])
                else:
                    e.append(items[i])
            D.append(e)
        D = np.array(D).astype(float)
        return D

    def build_kd_tree(self, x, left, right):
        if left > right:
            return
        else:
            if left == right:
                root = TreeNode()
                root.val = self.train[left]
            else:
                dim = divmod(x, self.k)[1]
                self.bubble_sort(dim, left, right)
                mid = math.modf((left + right) / 2)
                print(mid)
                midi = (int)(mid[1])
                if mid[0] >= 0.5:
                    midi += 1
                root = TreeNode()
                root.val = self.train[midi]
                root.left = self.build_kd_tree(x + 1, left, midi - 1)
                root.right = self.build_kd_tree(x + 1, midi + 1, right)
        return root

    def bubble_sort(self, x, left, right):
        for i in range(right - left + 1):  # 这个循环负责设置冒泡排序进行的次数
            for j in range(right - left - i):  # j为列表下标
                if self.train[left + j][x] > self.train[left + j + 1][x]:
                    t = self.train[left + j]
                    self.train[left + j] = self.train[left + j + 1]
                    self.train[left + j + 1] = t

    def execute(self):
        root = self.build_kd_tree(0, 0, len(self.train) - 1)
        print(self.train)
