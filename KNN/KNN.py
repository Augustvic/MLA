# k-Nearest Neighbor Algorithm
# Created by August
# 2018/10/27
# Source:1.李航. 统计学习方法[M]. 清华大学出版社, 2012.
#        2.https://www.jianshu.com/p/be23b3870d2e

from re import split
import numpy as np
import math
import copy


class TreeNode(object):
    def __init__(self, val=[], left=None, right=None, dim=None):
        self.val = val
        self.left = left
        self.right = right
        self.dim = dim


class KNN(object):
    def __init__(self):
        all_data_file_path = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\Iris.txt"
        self.all_data = self.load_data(all_data_file_path)
        # trainFilePath = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\train.txt"
        # self.train = self.load_data(trainFilePath)
        #test_file_path = r"C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\test.txt"
        #self.test = self.load_data(test_file_path)
        self.train = [np.array([2, 3]).astype(float),
                      np.array([5, 4]).astype(float),
                      np.array([9, 6]).astype(float),
                      np.array([4, 7]).astype(float),
                      np.array([8, 1]).astype(float),
                      np.array([7, 2]).astype(float)]
        self.test = [np.array([6, 3]).astype(float)]

        self.k = 2

    def load_data(self, filename):
        labels = {'Iris-setosa': '0', 'Iris-versicolor': '1', 'Iris-virginica': '2'}
        interval = ','
        with open(filename) as f:
            data = f.readlines()
        d = []
        for line in data:
            e = []
            items = split(interval, line.strip())
            for i in range(len(items)):
                if i == 4:
                    e.append(labels[items[i]])
                else:
                    e.append(items[i])
            d.append(e)
        d = np.array(d).astype(float)
        return d

    def build_kd_tree(self, x, left, right):     # Build KD Tree
        if left > right:
            return
        else:
            dim = divmod(x, self.k)[1]
            if left == right:
                root = TreeNode()
                root.val = self.train[left]
                root.dim = dim
            else:
                self.bubble_sort(dim, left, right)
                mid = math.modf((left + right) / 2)
                midi = (int)(mid[1])
                if mid[0] >= 0.5:
                    midi += 1
                root = TreeNode()
                root.val = self.train[midi]
                root.dim = dim
                root.left = self.build_kd_tree(x + 1, left, midi - 1)
                root.right = self.build_kd_tree(x + 1, midi + 1, right)
        return root

    def bubble_sort(self, x, left, right):     # Bubble Sort
        for i in range(right - left + 1):
            for j in range(right - left - i):
                if self.train[left + j][x] > self.train[left + j + 1][x]:
                    t = copy.deepcopy(self.train[left + j])
                    self.train[left + j] = copy.deepcopy(self.train[left + j + 1])
                    self.train[left + j + 1] = copy.deepcopy(t)

    def search_knn(self, root, e, neighbors):
        if root is None:
            return None
        dim = root.dim
        next_step_left = False
        if e[dim] < root.val[dim]:
            self.search_knn(root.left, e, neighbors)
        else:
            self.search_knn(root.right, e, neighbors)
            next_step_left = True
        self.update_neighbors(root, neighbors, e)
        if root.left is None and root.right is None:
            return None
        max_dist = self.max_dist_of_neighbors(neighbors, e)
        if abs(root.val[dim] - e[dim]) < max_dist:
            brother = root.left if next_step_left else root.right
            if brother is not None:
                self.search_knn(brother, e, neighbors)

    def dist(self, x, y):
        return np.sum((x - y) ** 2) ** 0.5

    def update_neighbors(self, root, neighbors, e):
        if len(neighbors) < self.k:
            neighbors.append(root.val)
        else:
            dist = self.dist(e, root.val)
            max_tmp = neighbors[0]
            max_dist = self.dist(e, max_tmp)
            for tmp in neighbors:
                curr_dist = self.dist(e, tmp)
                if curr_dist > max_dist:
                    max_tmp = tmp
                    max_dist = curr_dist
            if max_dist > dist:
                for i in range(len(neighbors)):
                    if neighbors[i] is max_tmp:
                        index = i
                del neighbors[index]
                neighbors.append(root.val)

    def max_dist_of_neighbors(self, neighbors, e):
        max_dist = self.dist(e, neighbors[0])
        for tmp in neighbors:
            curr_dist = self.dist(e, tmp)
            if curr_dist > max_dist:
                max_dist = curr_dist
        return max_dist

    def execute(self):
        root = self.build_kd_tree(0, 0, len(self.train) - 1)    # Build KD Tree
        for e in self.test:
            neighbors = []
            self.search_knn(root, e, neighbors)
            print(neighbors)
