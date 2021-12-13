import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)


class Model:
    test_data = []
    test_X = None
    test_Y = None
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

    def test(self, is_train=False):
        if self.W is None:
            print('Train Model First')
            return
        x = self.test_X if not is_train else self.train_X
        y = self.test_Y if not is_train else self.train_Y
        # value > 0 -> replacing with the class which represents y greater than 0
        y_pred = np.where(np.matmul(x, self.W) > 0, 1 if self.classes[1] > 0 else 2,
                          1 if self.classes[1] < 0 else 2)
        f = open('output.txt', 'w')
        count = 0
        for i in range(self.data_amount):
            if y_pred[i][0] != y[i][0]:
                count += 1
                f.write('smple_no: {}\tfeature_values: {}\tactual_class: {}\tpredicted_class: {}\n'
                        .format(i, [x[i][j] for j in range(1, len(x[i]))], y[i][0], y_pred[i][0]))
        accuracy = accuracy_score(y, y_pred)
        f.write('total_misclassified: {}\naccuracy:{}\n'.format(count, accuracy * 100))
        return accuracy, count

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

    def rnp(self, learning_rate=1):
        self.W = np.zeros((self.feature_amount + 1, 1))  # assigning zero weight

        for i in range(self.data_amount * 10):
            misclassified = False
            for j in range(self.data_amount):
                xw = np.matmul(self.train_X[j], self.W)
                y_class = self.classes[self.train_Y[j][0]]
                if y_class == -1 and xw >= 0:
                    self.W -= learning_rate * np.array(self.train_X[j]).reshape(-1, 1)
                    misclassified = True
                elif y_class == 1 and xw <= 0:
                    self.W += learning_rate * np.array(self.train_X[j]).reshape(-1, 1)
                    misclassified = True
            if not misclassified:
                break

    def pocket(self, learning_rate=1):
        temp_w = np.zeros((self.feature_amount + 1, 1))  # assigning zero weight

        correct_classify = 0
        for i in range(self.data_amount * 3):
            misclassified = []
            for j in range(self.data_amount):
                xw = np.matmul(self.train_X[j], temp_w)
                y_class = self.classes[self.train_Y[j][0]]
                if y_class == -1 and xw >= 0:
                    misclassified.append(np.array(self.train_X[j]).reshape(-1, 1))
                elif y_class == 1 and xw <= 0:
                    misclassified.append(-1 * np.array(self.train_X[j]).reshape(-1, 1))

            calc_correct_classify = self.data_amount - len(misclassified)
            if correct_classify < calc_correct_classify:
                if len(misclassified) == 9:
                    print(calc_correct_classify,correct_classify)
                correct_classify = calc_correct_classify
                self.W = temp_w
            if calc_correct_classify == self.data_amount:
                break

            # for item in misclassified:
            temp_w -= learning_rate * sum(misclassified)
