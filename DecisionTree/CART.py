
class CART(object):
    def __init__(self):
        train_file_path = r"C:\Users\August\PycharmProjects" \
                          r"\MachineLearningAlgorithm\Dataset\decisiontree.txt"
        self.train = self.load_data(train_file_path)

        self.last = 4

        self.label2num = {"age": 0, "job": 1, "house": 2, "money": 3}
        self.num2label = {0: "age", 1: "job", 2: "house", 3: "money"}

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
        print(sorted_count[-1][0])

    def min_gini(self, data):
        pass

    def build_tree(self, data):
        pass

    def execute(self):
        pass
