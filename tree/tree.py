class TN:
	def __init__(self, prediction, parent=None, left=None, right=None, id=None):
		self.parent = parent
		self._prediction = prediction # just initalize the root node with a bogus prediction. 
				# it's going to get deleted when you split it, which you must do to create a decision tree
		self.left  = left
		self.right = right
		self._is_leaf = True
		self.id = id # placed here for debugging purposes

	def size(self):
		if self.left == None:
			left_size = 0
		else:
			left_size = self.left.size()
		if self.right == None:
			right_size = 0
		else:
			right_size = self.right.size()
		return 1 + right_size + left_size

	def is_leaf(self):
		return self._is_leaf

	def height(self):
		if self.left == None:
			if self.right == None:
				return 0
			else:
				return 1 + self.right.height()
		if self.right == None:
			return 1 + self.left.height()
		else:
			return 1 + max(self.left.height(), self.right.height())

	def predict(self):
		return self._prediction

	def print_tree(self, indent_char ='....-....', indent_delta=3):
		'''Right branch details the case where the answer to the node's split query (feature x2 > 5, etc.)
			is true.'''
		def print_tree_1(indent,atree):
			if atree.is_leaf():
				# print('PREDICTING FOR NODE WITH ID: {}'.format(atree.id))
				# print(atree.is_leaf())
				print(indent * indent_char, atree.predict())
			else:
				print_tree_1(indent+indent_delta,atree.right)
				print(indent*indent_char+'(X{}>={}?)'.format(atree._feature_to_split, atree._feature_threshold))
				print_tree_1(indent+indent_delta,atree.left)
		print_tree_1(0, self) 
		print('height is {}'.format(self.height()))
		print('size is {}'.format(self.size()))

	def split(self, feature_to_split, feature_threshold, left_prediction, right_prediction):
		if not self.is_leaf():
			raise SplitError

		del self._prediction
		self._is_leaf = False
		self._feature_to_split = feature_to_split
		self._feature_threshold = feature_threshold

		if self.id is not None:
			self.left = TN(left_prediction, parent=self, id=(self.id * 2) + 1)
			self.right = TN(right_prediction, parent=self, id=(self.id * 2) + 2)
		else:
			self.left = TN(left_prediction, parent=self)
			self.right = TN(right_prediction, parent=self)







class SplitError(Exception):
	def __init__(self):
		print('Tried to split on tree node that was not a leaf.')
