
from sklearn.model_selection import train_test_split

c = []
filename = r'C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\Iris.txt'
out_train = open(r'C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\train.txt', 'w')
out_test = open(r'C:\Users\August\PycharmProjects\MachineLearningAlgorithm\Dataset\Iris\test.txt', 'w')

for line in open(filename):
    items = line.strip().split()
    c.append(items)

c_train, c_test = train_test_split(c, test_size=0.2)

traincount = 0
testcount = 0
for i in c_train:
    traincount += 1
    out_train.write(' '.join(i) + '\n')
for i in c_test:
    testcount += 1
    out_test.write(' '.join(i) + '\n')
print(traincount)
print(testcount)
