# CART Algorithm
# Created by August
# 2018/12/13
# Source:李航. 统计学习方法[M]. 清华大学出版社, 2012.
# Data Set: Watermelon2e.txt


from copy import deepcopy
import os.path
from time import strftime, localtime, time


class CART(object):
    def __init__(self):
#        train_file_path = r"C:\Users\August\PycharmProjects" \
#                          r"\MachineLearningAlgorithm\Dataset\decisiontree.txt"
#        self.train = self.load_data(train_file_path)
        train_file_path = r"C:\Users\August\PycharmProjects" \
                          r"\MachineLearningAlgorithm\Dataset\Watermelon\Watermelon2e_train.txt"
        self.train = self.load_data(train_file_path)

        test_file_path = r"C:\Users\August\PycharmProjects" \
                          r"\MachineLearningAlgorithm\Dataset\Watermelon\Watermelon2e_test.txt"
        self.test = self.load_data(test_file_path)

#        self.last = 4
        self.last = 6

        self.label2num = {"color": 0, "root": 1, "knock": 2, "pattern": 3, "umbilicus": 4, "touch": 5}
        self.num2label = {0: "color", 1: "root", 2: "knock", 3: "pattern", 4: "umbilicus", 5: "touch"}
#        self.label2num = {"age": 0, "job": 1, "house": 2, "money": 3}
#        self.num2label = {0: "age", 1: "job", 2: "house", 3: "money"}
        self.tag2num = {"no": 0, "yes": 1}
        self.num2tag = {0: "no", 1: "yes"}
        self.labels_yes_no2num = {"no": 1, "yes": 2}
        self.num2labels_yes_no = {1: "no", 2: "yes"}
#        self.labels_yes_no = [1, 2]
        self.labels_yes_no = [5]

    def load_data(self, filename):
        with open(filename) as f:
            data = f.readlines()
        d = []
        for line in data:
            e = []
            items = line.strip().split()
            for i in range(len(items)):
#                if i != 0:
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
                del gini[i][str(self.labels_yes_no2num["no"])]
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

        del_array = []
        for num in labels_values_remain.keys():
            count = 0
            value_dict = {}
            for tmp in data:
                count += 1
                if tmp[num] not in value_dict.keys():
                    value_dict[tmp[num]] = 0
                value_dict[tmp[num]] += 1
            for value in value_dict.keys():
                if value_dict[value] == count:
                    del_array.append(num)
                    break
        for num in del_array:
            del labels_values_remain[num]

        best_feature, best_feature_value = self.min_gini(data, labels_values_remain)

        root = {best_feature: {}}
        if best_feature in self.labels_yes_no:
            del labels_values_remain[best_feature]
        else:
            if len(labels_values_remain[best_feature]) == 0:
                del labels_values_remain[best_feature]
            else:
                labels_values_remain[best_feature].remove(best_feature_value)

        root[best_feature][best_feature_value] = self.build_tree(self.split_data(data, best_feature, best_feature_value, "false"), labels_values_remain)
        root[best_feature]["others"] = self.build_tree(self.split_data(data, best_feature, best_feature_value, "true"), labels_values_remain)
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

        if type(root) == str:
            return root
        label = list(root.keys())[0]
        value = watermelon_dict[self.num2label[label]]
        if type(root[label]["others"]) == str:
            if value not in root[label].keys():
                return root[label]["others"]
            else:
                return self.get_tag(root[label][value], watermelon)
        else:
            if value in root[label].keys():
                return root[label][value]
            else:
                return self.get_tag(root[label]["others"], watermelon)

    def hit(self, count_hit, count_test):
        return count_hit/count_test

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
        out_filename = "CART" + "@" + current_time + ".txt"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(out_path + out_filename, 'w') as f:
            f.writelines(predict)
        print("The predict result has been output to ..\Result")
        # evaluation
        print("HITS = " + str(self.hit(count_hit, count_test) * 100) + "%")

    def execute(self):
        labels_values_remain = {}
        for m in range(self.last):
            labels_values_remain[m] = []
        for info in self.train:
            for i in range(self.last):
                if info[i] not in labels_values_remain[i]:
                    labels_values_remain[i].append(info[i])
        root = self.build_tree(self.train, labels_values_remain)
        self.eval_predict(root)
