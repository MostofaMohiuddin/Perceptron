import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
SEED = 42
np.random.seed(SEED)


class Model:
    test_data = []
    test_X = []
    test_Y = []
    train_data = []
    train_X = None
    train_Y = None
    W = None
    feature_amount = 0
    data_amount = 0
    classes = {}

    def __init__(self, train_file, test_file):
        train_f = open(train_file, "r")
        test_f = open(test_file, "r")

        self.train_data, self.train_X, self.train_Y = self.read_data(train_f, True)
        self.test_data, self.test_X, self.test_Y = self.read_data(test_f, False)

        i = -1
        for key in Counter(self.train_Y[:, 0]):
            self.classes[key] = i
            i += 2

    def read_data(self, file, is_train):
        if is_train:
            self.feature_amount, classes, self.data_amount = [int(item) for item in file.readline().split()]
        if self.data_amount == 0:
            print('No data to read')
            return

        data = []

        for i in range(self.data_amount):
            line = [float(item) for item in file.readline().split()]
            data.append(line)
        np_data = np.array(data)
        x = np_data[:, :-1]
        x = np.insert(x, 0, 1, axis=1)
        y = np_data[:, -1].reshape(-1, 1).astype(int)
        return data, x, y

    def test(self):
        if self.W is None:
            print('Train Model First')
            return
        # value > 0 -> replacing with the class which represents y greater than 0
        y_pred = np.where(np.matmul(self.test_X, self.W) > 0, 1 if self.classes[1] > 0 else 2,
                          1 if self.classes[1] < 0 else 2)

        count = 0
        for i in range(self.data_amount):
            if y_pred[i][0] != self.test_Y[i][0]:
                count += 1
        print('error', count)
        print('Accuracy: ' , accuracy_score(self.test_Y, y_pred))

    def basic_perceptron(self, learning_rate=0.001):
        self.W = np.random.rand(self.feature_amount + 1, 1)  # assigning random weight

        misclassified = True

        while misclassified:
            misclassified = False
            grad = np.zeros((1, self.feature_amount + 1))
            for i in range(self.data_amount):
                delta = -self.classes[self.train_Y[i][0]]
                if delta * np.matmul(self.train_X[i], self.W) >= 0:
                    grad += delta * self.train_X[i]
                    misclassified = True

            self.W -= learning_rate * grad.T
