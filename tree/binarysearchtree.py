class TN:
    def __init__(self,value,left=None,right=None):
        self.value = value
        self.left  = left
        self.right = right

        
def size(atree):
    if atree == None:
        return 0
    else:
        return 1 + size(atree.left) + size(atree.right)


def height(atree):
    if atree == None:
        return -1
    else:
        return 1+ max(height(atree.left),height(atree.right))


def list_to_tree(alist):
    if alist == None:
        return None
    else:
        return TN(alist[0],list_to_tree(alist[1]),list_to_tree(alist[2])) 

    
def tree_to_list(atree):
    if atree == None:
        return None
    else:
        return [atree.value, tree_to_list(atree.left), tree_to_list(atree.right)]


def print_tree(atree,indent_char ='.',indent_delta=2):
    def print_tree_1(indent,atree):
        if atree == None:
            return None
        else:
            print_tree_1(indent+indent_delta,atree.right)
            print(indent*indent_char+str(atree.value))
            print_tree_1(indent+indent_delta,atree.left)
    print_tree_1(0,atree) 
    
    
def search_i(atree,value):
    while atree != None and atree.value != value:
        atree = atree.left if value < atree.value else atree.right
    return atree  # either None or the TN storing value
    

def search(atree,value):
    if atree == None:
        return None
    if value == atree.value:
        return atree
    elif value < atree.value:
        return search(atree.left,value)
    else: # value > atree.value
        return search(atree.right,value)
    

def add(atree,value):
    if atree == None:
        return TN(value)
    if value < atree.value:
        atree.left = add(atree.left,value)
        return atree
    elif value > atree.value:
        atree.right = add(atree.right,value)
        return atree
    else:
        return atree  # already in tree


def add_all(atree,values):
    for v in values:
        atree = add(atree,v)
    return atree


def copy(atree):
    if atree == None:
        return None
    else:
        return TN(atree.value, copy(atree.left), copy(atree.right)) 


def equal(t1,t2):
    if t1 == None or t2 == None:
        return t1 == t2
    else:
        return t1.value == t2.value and equal(t1.left,t2.left) and equal(t1.right,t2.right)


def generator_preorder(atree):
    if atree == None:
        return None
    else:
        yield atree.value
        for v in generator_preorder(atree.left):
            yield v
        for v in generator_preorder(atree.right):
            yield v


def generator_inorder(atree):
    if atree == None:
        return None
    else:
        for v in generator_inorder(atree.left):
            yield v
        yield atree.value
        for v in generator_inorder(atree.right):
            yield v


def generator_postorder(atree):
    if atree == None:
        return None
    else:
        for v in generator_postorder(atree.left):
            yield v
        for v in generator_postorder(atree.right):
            yield v
        yield atree.value


def generator_breadth_first(atree):
    if atree == None:
        return None
    to_yield = [atree]
    while to_yield != []:
        node = to_yield.pop(0)
        if node.left != None:
            to_yield.append(node.left)
        if node.right != None:
            to_yield.append(node.right)
        yield node.value

 
def random_height(n):
    from random import shuffle
    values = [i for i in range(n)]
    shuffle(values)
    return height(add_all(None,values))
     
     
if __name__ == '__main__':
    from random import shuffle
    import prompt,traceback
    def p(*args,**kargs):
        print(*args,**kargs)
        
    x = None
    x = add_all(x,[4,7,3,5,8,9,2,5])
    print_tree(x)
    while True:
        try:
            exec(prompt.for_string('Command'))
        except Exception:
            traceback.print_exc()   