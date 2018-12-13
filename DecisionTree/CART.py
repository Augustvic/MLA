# CART Algorithm
# Created by August
# 2018/12/13
# Source:李航. 统计学习方法[M]. 清华大学出版社, 2012.
# Data Set: Watermelon2e.txt

import math
from copy import deepcopy
import os.path
from time import strftime, localtime, time


class CART(object):
    def __init__(self):
        train_file_path = r"C:\Users\August\PycharmProjects" \
                          r"\MachineLearningAlgorithm\Dataset\decisiontree.txt"
        self.train = self.load_data(train_file_path)

        self.last = 4

        self.label2num = {"age": 0, "job": 1, "house": 2, "money": 3}
        self.num2label = {0: "age", 1: "job", 2: "house", 3: "money"}
        self.tag2num = {"no": 0, "yes":1}
        self.num2tag = {0:"no", 1:"yes"}
        self.labels_yes_no = [1, 2]

    def load_data(self, filename):
        with open(filename) as f:
            data = f.readlines()
        d = []
        for line in data:
            e = []
            items = line.strip().split()
            for i in range(len(items)):
                if i != 0:
                    e.append(items[i])
            d.append(e)
        return d

    def majority(self, data):
        count = {}
        for tag in data:
            if tag not in count.keys():
                count[tag] = 0
            count[tag] += 1
        sorted_count = sorted(count.items(), key=lambda item: item[1])
        return sorted_count

    def min_gini(self, data, labels_values_remain):
        gini_statistics = {}
        for i in labels_values_remain.keys():
            gini_statistics_label = {}
            for tmp in data:
                if i in labels_values_remain.keys() and tmp[i] in labels_values_remain[i]:
                    if tmp[i] not in gini_statistics_label .keys():
                        gini_statistics_label[tmp[i]] = {}
                    if tmp[self.last] not in gini_statistics_label[tmp[i]].keys():
                        gini_statistics_label[tmp[i]][tmp[self.last]] = 0
                    gini_statistics_label[tmp[i]][tmp[self.last]] += 1
            gini_statistics[i] = gini_statistics_label
        count = len(data)
        gini = {}
        for num in labels_values_remain.keys():
            dict = gini_statistics[num]
            for value in dict.keys():
                a = 0
                for label_value_value in dict[value].keys():
                    a += dict[value][label_value_value]
                b = 0
                for label_remain in dict.keys():
                    if label_remain != value:
                        if str(self.tag2num["yes"]) in dict[label_remain].keys():
                            b += dict[label_remain][str(self.tag2num["yes"])]
                if num not in gini.keys():
                    gini[num] = {}
                if str(self.tag2num["yes"]) in dict[str(value)].keys():
                    s = dict[str(value)][str(self.tag2num["yes"])]
                else:
                    s = 0
                gini[num][value] = a / count * (2 * s / a * (1 - s / a)) + (count - a) / count * (2 * b / (count - a) * (1 - b / (count - a)))

        min_label = 0
        min_label_value = '0'
        min_gini = 100
        for i in gini.keys():
            if i in self.labels_yes_no:
                del gini[i][str(self.tag2num["no"])]
            for j in gini[i].keys():
                if min_gini > gini[i][j]:
                    min_label = i
                    min_label_value = j
                    min_gini = gini[i][j]
        return [min_label, min_label_value]

    def split_data(self, data, feature, value, others):
        new_data = []
        if others == "false":
            for tmp in data:
                if tmp[feature] == value:
                    new_data.append(deepcopy(tmp))
        else:
            for tmp in data:
                if tmp[feature] != value:
                    new_data.append(deepcopy(tmp))
        return new_data

    def build_tree(self, data, labels_values_remain):
        tags = []
        for tmp in data:
            tags.append(tmp[self.last])
        if len(set(tags)) == 1:
            return list(tags)[0]
        if len(labels_values_remain) == 0:
            return self.majority(tags)

        best_feature, best_feature_value = self.min_gini(data, labels_values_remain)

        root = {best_feature: {}}
        if best_feature in self.labels_yes_no:
            del labels_values_remain[best_feature]
        else:
            if len(labels_values_remain[best_feature]) == 0:
                del labels_values_remain[best_feature]
            else:
                del labels_values_remain[best_feature][int(best_feature_value)]

        root[best_feature][best_feature_value] = self.build_tree(self.split_data(data, best_feature, best_feature_value, "false"), labels_values_remain)
        root[best_feature]["others"] = self.build_tree(self.split_data(data, best_feature, best_feature_value, "true"), labels_values_remain)
        return root

    def execute(self):
        labels_values_remain = {0: [], 1: [], 2: [], 3: []}
        for info in self.train:
            for i in range(self.last):
                if info[i] not in labels_values_remain[i]:
                    labels_values_remain[i].append(info[i])
        root = self.build_tree(self.train, labels_values_remain)
