import model
import numpy as np
print("Perceptron")

f = open("trainLinearlySeparable.txt", "r")

print(f.readline())
model1 = model.Model('trainLinearlySeparable.txt','testLinearlySeparable.txt')
model1.basic_perceptron()
print(np.dot(model1.train_X[1], model1.train_X[0]))
print(model1.W)
model1.test()