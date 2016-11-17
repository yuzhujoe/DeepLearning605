# some useful functions
import numpy as np
from xman import *


# some useful functions
# declare all operations here first

class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)

    @staticmethod
    def relu(a):
        return XManFunctions.registerDefinedByOperator('relu',a)

    @staticmethod
    def softMax(x):
        return XManFunctions.registerDefinedByOperator('softMax',x)

    @staticmethod
    def crossEnt(x,t):
        return XManFunctions.registerDefinedByOperator('crossEnt',x,t)

    @staticmethod
    def mean(x):
        return XManFunctions.registerDefinedByOperator('mean',x)

    @staticmethod
    def sigmoid(x):
        return XManFunctions.registerDefinedByOperator('sigmoid',x)

    @staticmethod
    def tanh(x):
        return XManFunctions.registerDefinedByOperator('tanh',x)

    @staticmethod
    def elementMul(x1,x2):
        return XManFunctions.registerDefinedByOperator('elementMul',x1,x2)

    # TODO add other operation registers
# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments


EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'mul':      lambda x1,x2: np.dot(x1,x2),
    'square':   np.square,
    'crossEnt': lambda  x,t: -1 * np.sum(t * np.log(x),axis= 1),
    'softMax':  lambda x1 : np.exp(x1 - np.max(x1,axis=1,keepdims=True)) / np.sum(np.exp(x1 - np.max(x1,axis=1,keepdims= True)),axis=1,keepdims=True),
    'relu':     lambda x1 : x1 * (x1 >= 0),
    'mean':     np.mean,
    'sigmoid' : lambda x1 : 1/(1 + np.exp(-1 * x1)),
    'tanh':     np.tanh,
    'elementMul': lambda x1,x2:x1 * x2
    # TODO other operations
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
# 
# NOTE: Autograd has an optimization where if it finds a softMax op
# followed by crossEnt op, it combines the backward pass for both. So
# you only need to implement the BP_FUNS for the combined operation 
# crossEnt-softMax below.

def _derivAdd(delta,x1):
    if delta.shape!=x1.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x1.shape[0]:
            raise ValueError("Dimension Mismatch")
        return delta.sum(axis=0) #we sum the gradients over the batch
    else: return delta


def _derivRelu(delta,x1):
    x1[x1>=0] = 1
    x1[x1 < 0] = 0
    return delta * x1


def _derivCrossEntSoftMaxX(delta, x , t):
    softMax = np.exp(x - np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x - np.max(x,axis=1,keepdims= True)),axis=1,keepdims=True)

    return (softMax - t) * delta

def _derivCrossEntSoftMaxT(delta, x , t):
    softMax = np.exp(x - np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x - np.max(x,axis=1,keepdims= True)),axis=1,keepdims=True)
    return  -np.log(softMax) * delta

BP_FUNS = {
    'add':              [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'subtract':         [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'mul':         [lambda delta,out,x1,x2: np.dot(delta,x2.transpose()),    lambda delta,out,x1,x2 : np.dot(x1.transpose(),delta)],
    'sigmoid':      [lambda delta,out,x1 : delta * (out * (1 - out))],
    'tanh':         [lambda delta,out,x1 :delta * (1 - out * out)],
    'elementMul':   [lambda delta,out,x1,x2 :delta * x2, lambda delta,out,x1,x2: delta * x1],
    'square':           [lambda delta,out,x : delta * 2.0 * x],
    'crossEnt-softMax': [lambda delta,out,x,t : _derivCrossEntSoftMaxX(delta,x,t),   lambda delta,out,x,t : _derivCrossEntSoftMaxT(delta,x,t)],
    'relu': [lambda delta,out,x: _derivRelu(delta,x)],
    'mean': [lambda delta,out,x: delta * np.ones((x.shape[0],1)) / x.size]
    # TODO other operations
    }