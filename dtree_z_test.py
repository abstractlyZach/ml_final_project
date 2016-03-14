from mltools import dtree_z
import numpy as np

'''
test = np.genfromtxt('data/zach_test_data.txt', delimiter=',')
X = test[:, :-1]
Y = test[:, -1]
print(X)
print(Y)
a = dtree_z.treeRegress()
a.train(X, Y, 2)
a.print_tree()

feature = 2
go_left = X[:, feature] % 3 == 0
print(go_left)
print(X[go_left])
print(X[go_left, feature])
print(Y[go_left])

print('\n\n')
print('====' * 8)
print('beginning IRIS.txt\n\n')

data = np.genfromtxt('data/iris.txt')
X = data[:, :-1]
Y = data[:, -1:]


learner = dtree_z.treeRegress()
learner.train(X, Y, 100)
learner.print_tree()
learner.predict(np.asarray([[1,1,1,1],[2,2,2,2],[3,3,3,3]]))
'''


import mltools as ml
import numpy as np
import matplotlib.pyplot as plt
import mltools.dtree_z as dt

data = np.genfromtxt('data/curve80.txt')
X = data[:, :-1]
Y = data[:,-1:]

learner = dt.treeRegress()

learner.train(X, Y, 2)

lines = plt.plot(X, Y, 'r.')