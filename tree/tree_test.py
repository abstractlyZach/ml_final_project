# testing tree.py
import tree

decision_tree = tree.TN(3, id=0)
decision_tree.print_tree()
print()
print()

current_node = decision_tree
current_node.split(2, 5.4, 2, 6)

decision_tree.print_tree()
print()
print()

current_node = current_node.left
current_node.split(99, 37.8, 55, 93)

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