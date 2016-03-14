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
				# priority is determined by the information gain.

	def train(self, X, Y, maxLeaves, minParent=2, nFeatures=None):
		'''
		Trains the learner on X and Y using maxLeaves to limit complexity.
		'''
		self.tree = tree.TN(0) # initialize tree
		self.node_information_gain.add([self.tree] + self.__best_feature(X, Y)) # add best feature to pq
		while self.tree.leaves() < maxLeaves and not self.node_information_gain.is_empty(): # loop until the number of leaves reaches maxLeaves
			# pop the queue to obtain the node with the most information gain
			current_node, best_val, best_feature, best_thresh = self.node_information_gain.remove()
			if best_val == 0: # no best value to split on
				continue
			# obtain prediction values for each of the leaves-to-be
			Yhat_left, Yhat_right = self.__leaf_values(X, Y, best_thresh, best_feature)
			# figure out which rows of X get sent into the left and right nodes
			left_rows = current_node.rows & (X[:, best_feature] < best_thresh)
			right_rows = current_node.rows & (X[:, best_feature] > best_thresh)
			# split the current node
			current_node.split(best_feature, best_thresh, Yhat_left, Yhat_right, left_rows, right_rows)
			# add the new leaves to the priority queue
			if (Y[left_rows] == Y[left_rows][0]).all == True: # each entry in Y is the same as Y
				# can't split the data any more
				if (Y[right_rows] == Y[right_rows][0]).all == True: # each entry in Y is the same as Y
					pass # can't split the data any more
				else:
					self.node_information_gain.add([current_node.right] + self.__best_feature(X[right_rows], Y[right_rows]))
			elif (Y[right_rows] == Y[right_rows][0]).all == True: # each entry in Y is the same as Y
				self.node_information_gain.add([current_node.right] + self.__best_feature(X[left_rows], Y[left_rows]))

			else:
				self.node_information_gain.add([current_node.left] + self.__best_feature(X[left_rows], Y[left_rows]))
				self.node_information_gain.add([current_node.right] + self.__best_feature(X[right_rows], Y[right_rows]))

	def predict(self, X):
		'''
		predicts Yhat for each value in X and returns an array of those prediction values
		'''
		to_return = np.array()
		for value in X:
			to_return.append(self.tree.predict(value))
		return to_return



	def __best_feature(self, X, Y):
		'''
		finds best feature to split on in X and returns a list of important info
		'''
		rows, num_features = mat(X).shape
		max_variance_reduction = 0
		best_val = 0
		best_feature = 0
		best_thresh = 0
		for feature in range(num_features): # examine each feature
			feature_data = []
			for index, x_value in enumerate(X[:, feature]): # attach each value to its index
				feature_data.append((index, x_value))
			feature_data.sort(key=lambda x: x[1]) # sort values
			for split_index in range(len(feature_data) - 1): # create splits between data
				split1 = feature_data[:split_index + 1]
				split2 = feature_data[split_index + 1:]
				if split1[-1][1] == split2[0][1]: # if the two values that are to be split are equal
					continue
				variance_reduction = self.__variance_reduction(split1, split2, Y) 
				if variance_reduction > max_variance_reduction: 
				# if the weighted variance is the least, save the weighted_variance, feature, and index to split on that feature
					max_variance_reduction = variance_reduction
					best_val = variance_reduction
					best_feature = feature
					best_thresh = (feature_data[split_index][1] + feature_data[split_index + 1][1]) / 2
		return [best_val, best_feature, best_thresh]

	def __leaf_values(self, X, Y, best_thresh, best_feature):
		'''
		return a tuple with the predictions for left and right leaves after the split
		'''
		go_left = X[:, best_feature] < best_thresh
		go_right = X[:, best_feature] > best_thresh
		Yhat_left = np.mean(Y[go_left])
		Yhat_right = np.mean(Y[go_right])
		return (Yhat_left, Yhat_right)

	def __variance_reduction(self, split1, split2, Y):
		'''measures the weighted variance reduction'''
		original_var = np.var(Y)
		# calculate split1's variance
		total = 0
		for index, X_value in split1:
			total += Y[index]
		avg = total / len(split1)
		total = 0
		for index, X_value in split1:
			total += (X_value - avg) ** 2
		split1_var = total / len(split1)
		# calculate split1's variance
		total = 0
		for index, X_value in split2:
			total += Y[index]
		avg = total / len(split2)
		total = 0
		for index, X_value in split2:
			total += (X_value - avg) ** 2
		split2_var = total / len(split2)
		return (len(split1) / len(Y)) * (original_var - split1_var) + \
					(len(split2) / len(Y)) * (original_var - split2_var) 

	def print_tree(self):
		self.tree.print_tree()

