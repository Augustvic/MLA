# C4.5 Algorithm
# Created by August
# 2018/12/11
# Source:李航. 统计学习方法[M]. 清华大学出版社, 2012.
# Data Set: Watermelon2e.txt


import math
from copy import deepcopy
import os.path
from time import strftime, localtime, time


class C45(object):
    def __init__(self):
        train_file_path = r"C:\Users\August\PycharmProjects" \
                          r"\MachineLearningAlgorithm\Dataset\Watermelon\Watermelon2e_train.txt"
        self.train = self.load_data(train_file_path)
        test_file_path = r"C:\Users\August\PycharmProjects" \
                         r"\MachineLearningAlgorithm\Dataset\Watermelon\Watermelon2e_test.txt"
        self.test = self.load_data(test_file_path)

        self.last = 6

        self.label2num = {"color": 0, "root": 1, "knock": 2, "pattern": 3, "umbilicus": 4, "touch": 5}
        self.num2label = {0: "color", 1: "root", 2: "knock", 3: "pattern", 4: "umbilicus", 5: "touch"}

    def load_data(self, filename):
        with open(filename) as f:
            data = f.readlines()
        d = []
        for line in data:
            e = []
            items = line.strip().split()
            for i in range(len(items)):
                if i != 0 and i != 7 and i != 8:
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
        print(sorted_count[-1][0])

    def max_information_gain(self, data, labels):
        max_gain_ratio = -1
        best_feature = -1
        # calculate the H(D)
        count = {}
        for tmp in data:
            tag = tmp[self.last]
            if tag not in count.keys():
                count[tag] = 0
            count[tag] += 1
        hd = 0
        for key in count:
            hd -= count[key] / len(data) * math.log2(count[key] / len(data))
        # calculate information gains
        labels_list = []
        for m in labels:
            labels_list.append(self.label2num[m])
        for i in labels_list:
            feature = {}
            feature_tags = {}
            for tmp in data:
                fea = tmp[i]
                if fea not in feature.keys():
                    feature[fea] = 0
                feature[fea] += 1
                if fea not in feature_tags.keys():
                    feature_tags[fea] = {}
                tag = tmp[self.last]
                if tag not in feature_tags[fea]:
                    feature_tags[fea][tag] = 0
                feature_tags[fea][tag] += 1

            hda = 0
            for fea in feature.keys():
                for tag in feature_tags[fea]:
                    hda -= feature[fea] / len(data) * math.log2(feature_tags[fea][tag] / feature[fea])
            information_gain_ratio = (hd - hda) / hd

            if information_gain_ratio > max_gain_ratio:
                max_gain_ratio = information_gain_ratio
                best_feature = self.num2label[i]

        return best_feature

    def split_data(self, data, axis, value):
        new_data = []
        for tmp in data:
            if tmp[self.label2num[axis]] == value:
                new_data.append(deepcopy(tmp))
        return new_data

    def build_tree(self, data, labels):
        tags = []
        for tmp in data:
            tags.append(tmp[self.last])
        if len(set(tags)) == 1:
            return list(tags)[0]
        if len(labels) == 0:
            return self.majority(tags)

        best_feature = self.max_information_gain(data, labels)
        root = {best_feature: {}}
        labels.remove(best_feature)
        feature_values = []
        for tmp in data:
            i = self.label2num[best_feature]
            feature_values.append(tmp[i])
        unique_value = set(feature_values)
        for value in unique_value:
            sub_labels = deepcopy(labels)
            root[best_feature][value] = self.build_tree(self.split_data(data, best_feature, value), sub_labels)

        return root

    def get_tag(self, root, watermelon):
        color, root_w, knock, pattern, umbilicus, touch, tag = watermelon
        watermelon_dict = {}
        watermelon_dict["color"] = color
        watermelon_dict["root"] = root_w
        watermelon_dict["knock"] = knock
        watermelon_dict["pattern"] = pattern
        watermelon_dict["umbilicus"] = umbilicus
        watermelon_dict["touch"] = touch
        watermelon_dict["tag"] = tag
        tmp1 = root
        while not isinstance(tmp1, str):
            label = list(tmp1.keys())
            val = watermelon_dict[label[0]]
            tmp2 = tmp1.get(label[0])
            tmp1 = tmp2.get(val)
        return tmp1

    def hit(self, count_hit, count_test):
        return count_hit / count_test

    def eval_predict(self, root):
        # output to file
        predict = []
        count_hit = 0
        count_test = 0
        for watermelon_test in self.test:
            count_test += 1
            color, root_w, knock, pattern, umbilicus, touch, tag = watermelon_test
            pred = self.get_tag(root, watermelon_test)
            if pred == tag:
                count_hit += 1
            predict.append(str(color) + ',' + str(root_w) + ',' + str(knock) + ',' + str(pattern) + ','
                           + str(umbilicus) + ',' + str(touch) + ',' + str(pred) + ',' + str(tag) + '\n')
        out_path = "../Result/"
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_filename = "C45_Predict_for_Watermelon" + "@" + current_time + ".txt"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(out_path + out_filename, 'w') as f:
            f.writelines(predict)
        print("The predict result has been output to ..\Result")
        # evaluation
        print("HITS = " + str(self.hit(count_hit, count_test) * 100) + "%")

    def execute(self):
        labels = ["color", "root", "knock", "pattern", "umbilicus", "touch"]
        data = deepcopy(self.train)
        root = self.build_tree(data, labels)
        self.eval_predict(root)