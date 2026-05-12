import numpy as np


def convert_to_new_format(stack):
    if (stack[0].dtype == np.float32):
        new_stack=[]
        for f in range(len(stack)):
            new_stack.append(np.array(stack[f]*10000,dtype = np.uint16))
        return new_stack
    return stack
