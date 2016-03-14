# testing tree.py
import tree

decision_tree = tree.TN(3, id=0)
decision_tree.print_tree()
assert decision_tree.leaves() == 1
print()
print()

current_node = decision_tree
current_node.split(2, 5.4, 2, 6)
assert decision_tree.leaves() == 2

decision_tree.print_tree()
print()
print()

current_node = current_node.left
current_node.split(99, 37.8, 55, 93)
assert decision_tree.leaves() == 3

decision_tree.print_tree()
print()
print()

current_node = current_node.parent.right
current_node.split(77, 334.85, 44, 22)

decision_tree.print_tree()
print()
print()


current_node = current_node.left
current_node.split(99, 37.8, 55, 93)
decision_tree.print_tree()
print()
print()

current_node = current_node.left
current_node.split(99, 37.8, 55, 93)
current_node = current_node.left
current_node.split(99, 37.8, 55, 93)
current_node = current_node.left
current_node.split(99, 37.8, 55, 93)
current_node = current_node.left
current_node.split(99, 37.8, 55, 93)

decision_tree.print_tree()

current_node.print_tree()

decision_tree = tree.TN(5, id=0)
current_node = decision_tree
current_node.split(0, 2.5, 2, 4)
current_node = current_node.left
current_node.split(0, 1.5, 1, 2)
current_node = current_node.parent.right
current_node.split(0, 3.5, 3, 4)

decision_tree.print_tree()

assert decision_tree.predict([1, 2, 3]) == 1
assert decision_tree.predict([2,3,4]) == 2
assert decision_tree.predict([5,5,5]) == 4

assert decision_tree.leaves() == 4

