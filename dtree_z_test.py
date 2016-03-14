from mltools import dtree_z
import numpy as np

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

