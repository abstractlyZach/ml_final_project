# node-based regression tree
from . import tree
import numpy as np
from .base import regressor
from . import priorityqueue
from numpy import asmatrix as mat
from numpy import asarray as arr



class treeRegress(regressor):
	def __init__(self):
		self.node_information_gain = priorityqueue.PriorityQueue(key=lambda x: x[1]) 
			# priority queue that takes in tuples where the first item is a leaf node and the second
				# item is the information gained by splitting at that node. Third is feature to split on and
				# fourth is at which value of the feature to split on.

	def train(self, X, Y, maxLeaves, minParent=2, nFeatures=None):
		self.tree = tree.TN(0)
		self.node_information_gain.add([self.tree] + self.__best_feature(X, Y))
		leaves = set()
		leaves.add(self.tree)
		while self.tree.leaves() < maxLeaves:
			current_node, best_val, best_feature, best_thresh = self.node_information_gain.remove()
			leaves.remove(current_node)
			Yhat_left, Yhat_right = self.__leaf_values(X, Y, best_thresh, best_feature)
			left_rows = current_node.rows & (X[:, best_feature] < best_thresh)
			right_rows = current_node.rows & (X[:, best_feature] > best_thresh)
			current_node.split(best_feature, best_thresh, Yhat_left, Yhat_right, left_rows, right_rows)
			self.node_information_gain.add([current_node.left] + self.__best_feature(X[left_rows], Y[left_rows]))
			leaves.add(current_node.left)
			leaves.add(current_node.right)



			# print(feature_data)

		# while self.tree.leaves() < maxLeaves:
		# 	best_val = np.inf

	def __best_feature(self, X, Y):
		rows, num_features = mat(X).shape
		min_weighted_variance = np.inf
		for feature in range(num_features): # examine each feature
			feature_data = []
			for index, x_value in enumerate(X[:, feature]): # attach each value to its index
				feature_data.append((index, x_value))
			feature_data.sort(key=lambda x: x[1]) # sort values
			for split_index in range(len(feature_data) - 1): # create splits between data
				split1 = feature_data[:split_index + 1]
				split2 = feature_data[split_index + 1:]
				weighted_variance = self.__weighted_variance(split1, split2, Y) 
				if weighted_variance < min_weighted_variance: 
				# if the weighted variance is the least, save the weighted_variance, feature, and index to split on that feature
					best_val = weighted_variance
					best_feature = feature
					best_thresh = (feature_data[split_index][1] + feature_data[split_index + 1][1]) / 2
		return [best_val, best_feature, best_thresh]

	def __leaf_values(self, X, Y, best_thresh, best_feature):
		go_left = X[:, best_feature] < best_thresh
		go_right = X[:, best_feature] > best_thresh
		Yhat_left = np.mean(Y[go_left])
		Yhat_right = np.mean(Y[go_right])
		return (Yhat_left, Yhat_right)

	def __weighted_variance(self, split1, split2, Y):
		# what do i do here
		return 0

	def print_tree(self):
		self.tree.print_tree()

