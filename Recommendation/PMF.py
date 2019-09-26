# Probabilistic Matrix Factorization Algorithm
# Created by August
# 2018/10/13
# Source:1.Salakhutdinov R, Mnih A. Probabilistic Matrix Factorization[C]// International Conference on Neural
#          Information Processing Systems. Curran Associates Inc. 2007:1257-1264.
#        2.https://blog.csdn.net/shenxiaolu1984/article/details/50372909


from re import split
import numpy as np
from collections import defaultdict
from Evaluation import Evaluate
from time import strftime, localtime, time
import os.path


class PMF(object):
    def __init__(self):
        print("-------------------------------PMF---------------------------------")
        # get all rating records, training data set and testing data set from local file
        ratings_file_path = r"C:\Users\August\PycharmProjects\RecommendAlgorithm\dataset\doubanTest\ratings.txt"
        self.ratings = self.load_data(ratings_file_path)
        train_file_path = r"C:\Users\August\PycharmProjects\RecommendAlgorithm\dataset\doubanTest\train.txt"
        self.train = self.load_data(train_file_path)
        test_file_path = r"C:\Users\August\PycharmProjects\RecommendAlgorithm\dataset\doubanTest\test.txt"
        self.test = self.load_data(test_file_path)

        # definite parameters in this algorithm
        self.factors = 10
        self.iter = 20
        self.learningRate = 0.05
        self.lambda_u = 0.01
        self.lambda_v = 0.001
        self.P = []
        self.Q = []
        self.test_uir = []
        self.user = {}
        self.item = {}
        self.ratings_ui = defaultdict(dict)
        self.test_ui = defaultdict(dict)
        self.predict = defaultdict(dict)

    def load_data(self, filename):
        with open(filename) as f:
            data = f.readlines()
        D = []
        for line in data:
            e = []
            items = split(' ', line.strip())
            e.append(items[0])
            e.append(items[1])
            e.append(items[2])
            e = np.array(e, dtype=float)
            D.append(e)
        return D

    def init_algorithm(self):
        # set the parameters up
        self.factors = 40
        self.iter = 10
        self.learningRate = 0.01
        self.lambda_u = 0.01
        self.lambda_v = 0.01
        print("The parameters in this algorithm are as following:")
        print("Number of potential factors:  " + str(self.factors))
        print("Maximum number of iterations: " + str(self.iter))
        print("Learning rate:                " + str(self.learningRate))
        print("Lambda U:                     " + str(self.lambda_u))
        print("Lambda V:                     " + str(self.lambda_v))
        print("-------------------------------MF---------------------------------")

        print("Init the algorithm...")
        # fill the user array and the item array
        for record in self.ratings:
            user_name, item_name, rating = record
            if user_name not in self.user.values():
                self.user[user_name] = len(self.user)
            if item_name not in self.item.values():
                self.item[item_name] = len(self.item)
            self.ratings_ui[user_name][item_name] = rating
        # initialize matrix P and matrix Q randomly
        self.P = np.random.rand(len(self.user), self.factors)/2
        self.Q = np.random.rand(len(self.item), self.factors)/2
        print("Init successfully!")
        print("-------------------------------MF---------------------------------")

    def run_algorithm(self):
        print("Run the algorithm...")
        # iteration process of Matrix Factorization Algorithm
        iter_curr = 0
        while iter_curr < self.iter:
            loss = 0
            for record in self.train:
                user_name, item_name, rating = record
                user_id = self.user[user_name]
                item_id = self.item[item_name]
                error = rating - self.P[user_id].dot(self.Q[item_id])
                loss += error
                p = self.P[user_id]
                q = self.Q[item_id]
                self.P[user_id] += self.learningRate * (error * q - self.lambda_u * p)
                self.Q[item_id] += self.learningRate * (error * p - self.lambda_v * q)
            iter_curr += 1
            print("Iteration %s: Loss %s" % (str(iter_curr), loss))
        print("Finished!")
        print("-------------------------------MF---------------------------------")

    def evaluate(self):
        out_predict = []
        for record in self.test:
            user_name, item_name, rating = record
            user_id = self.user[user_name]
            item_id = self.item[item_name]
            self.test_ui[user_name][item_name] = rating
            e = []
            e.append(user_name)
            e.append(item_name)
            e.append(rating)
            e = np.array(e, dtype=float)
            self.test_uir.append(e)
            self.predict[user_name][item_name] = round(self.P[user_id].dot(self.Q[item_id]), 0)

            out_predict.append(str(user_name) + '   ' + str(item_name) + '   ' + str(rating) + '   '
                               + str(self.predict[user_name][item_name]) + '\n')

        # output to file
        out_path = "../Result/"
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_filename = "PMF_Predict_for_Rating" + "@" + current_time + ".txt"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(out_path + out_filename, 'w') as f:
            f.writelines(out_predict)
        print("The predict result has been output to ..\Dataset\Result")

        evaluate = Evaluate(self.test_uir, self.predict)
        rmse = evaluate.RMSE()
        mae = evaluate.MAE()
        print("Evaluate:")
        print("RMSE: %s, MAE: %s" % (rmse, mae))

    def execute(self):
        self.init_algorithm()
        self.run_algorithm()
        self.evaluate()
