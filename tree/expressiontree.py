from math import  *  #use functions

class TN:
    def __init__(self,value,left=None,right=None):
        self.value = value
        self.left  = left
        self.right = right

        
def print_tree(atree,indent_char ='.',indent_delta=2):
    def print_tree_1(indent,atree):
        if atree == None:
            return None
        else:
            print_tree_1(indent+indent_delta,atree.right)
            print(indent*indent_char+str(atree.value))
            print_tree_1(indent+indent_delta,atree.left)
    print_tree_1(0,atree) 
    
    
def evaluate(etree):
    if etree.left == None and etree.right == None:
        return eval(str(etree.value))
    elif etree.left == None:
        if etree.value in ['+','-','*','/','//','**']:
            return eval(etree.value + str(evaluate(etree.right)))
        else:
            return eval(etree.value+'('+str(evaluate(etree.right))+')')
    else:
        return eval(str(evaluate(etree.left)) + etree.value + str(evaluate(etree.right)))
 
    
def infix(etree):
    if etree.left == None and etree.right == None:
        return '('+str(etree.value)+')'
    elif etree.left == None:
        return '('+etree.value+str(infix(etree.right))+')'
    else:
        return '('+str(infix(etree.left))+etree.value+str(infix(etree.right))+')'


def postfix(etree):
    if etree.left == None and etree.right == None:
        return str(etree.value)
    elif etree.left == None:
        return str(postfix(etree.right)) + ' ' + etree.value
    else:
        return str(postfix(etree.left)) + ' ' + str(postfix(etree.right)) + ' ' + etree.value

# works only for binary operators and assumes all operators are left associative (** is not)
def parse_infix(expression):
    operators = '+ - * / // ***'.split(' ')
    prec = {'+' : 1, '-' : 2, '*' : 2, '/' : 2, '//' : 2, '**' : 3}
    exp_stack = []
    op_stack = []
    for t in expression.split(' '):
        if t not in operators and t not in '()':
            exp_stack.append(TN(t,None,None))
        elif t == '(':
            op_stack.append(t)
        elif t == ')':
            while True:
                assert op_stack,'parse_infix: extra )'
                op = op_stack.pop(-1)
                if op == '(':
                    break
                right = exp_stack.pop(-1)
                left  = exp_stack.pop(-1)
                exp_stack.append(TN(op,left,right))
        elif t in operators:
            while op_stack:
                op = op_stack[-1]
                if op == '(' or prec[t] > prec[op]:
                    break
                op_stack.pop(-1)
                right = exp_stack.pop(-1)
                left  = exp_stack.pop(-1)
                exp_stack.append(TN(op,left,right))
            op_stack.append(t)
    for t in reversed(op_stack):
        assert t != '(', 'parse_infix: extra ('
        right = exp_stack.pop(-1)
        left  = exp_stack.pop(-1)
        exp_stack.append(TN(t,left,right))
    return exp_stack[0]
                
                
                
if __name__ == '__main__':
    from random import shuffle
    import prompt,traceback
    def p(*args,**kargs):
        print(*args,**kargs)
    t = \
        TN('/',
           TN('+',
              TN('-',
                 None,
                 TN('b',None,None)
                ),
              TN('sqrt',
                  None,
                  TN('-',
                     TN('**',
                        TN('b',None,None),
                        TN(2,None,None)
                        ),
                     TN('*',
                        TN('*',
                           TN('4',None,None),
                           TN('a',None,None)
                          ),
                        TN('c',None,None)
                       )
                     )
                   )
                ),
           TN('*',
              TN(2,None,None),
              TN('a',None,None)
             )  
           )

    print_tree(t)
    a = 1
    b = 2
    c = 1
    print(evaluate(t))
    
    print(infix(t))
    print(postfix(t))
    

    
    
    while True:
        try:
            exec(prompt.for_string('Command'))
        except Exception:
            traceback.print_exc()