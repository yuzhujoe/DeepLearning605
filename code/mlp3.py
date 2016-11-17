"""
Multilayer Perceptron for character level entity classification
"""
import argparse
from copy import deepcopy
from math import sqrt

import numpy as np
from xman import *
from utils import *
from autograd import *

epsilon = 10e-4
np.random.seed(0)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.layer_sizes_list = layer_sizes
        self.in_size = layer_sizes[0]
        self.hid_size1 = layer_sizes[1]
        self.out_size = layer_sizes[2]
        self.my_xman = self._build() # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        # print "in_size: "+ str(self.in_size)
        w1scale = sqrt(6.0/(self.in_size + self.hid_size1))
        w2scale = sqrt(6.0/(self.hid_size1 + self.out_size))

        # x.x = f.input(name='x', default = np.ones((1,self.in_size))/self.in_size)
        x.x = f.input(name='x', default = np.random.rand(1,self.in_size))
        x.y = f.input(name='y', default = np.ones((1,self.out_size))/self.out_size)
        x.W1 = f.param(name='W1', default = np.random.uniform(-1*w1scale,w1scale,(self.in_size,self.hid_size1)))
        x.b1 = f.param(name='b1', default = np.random.uniform(0,0.1, (self.hid_size1,)))
        x.W2 = f.param(name='W2', default = np.random.uniform(-1*w2scale,w2scale,(self.hid_size1,self.out_size)))
        x.b2 = f.param(name='b2', default = np.random.uniform(0,0.1,(self.out_size,)))

        x.o1 = f.relu(f.add(f.mul(x.x,x.W1),x.b1))
        x.o2 = f.relu(f.add(f.mul(x.o1,x.W2),x.b2))

        x.outputs = f.softMax(x.o2)
        x.loss = f.mean(f.crossEnt(x.outputs,x.y))
        #TODO define your model here
        return x.setup()

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    # build
    print "building mlp..."
    print "V size: " + str(mb_train.num_chars)
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])

    # compute LHS
    gradient_check_value_dict = mlp.my_xman.inputDict()
    gradient_check_wengert_list = mlp.my_xman.operationSequence(mlp.my_xman.loss)
    gradient_check_ad = Autograd(mlp.my_xman)
    gradient_check_value_dict = gradient_check_ad.eval(gradient_check_wengert_list,valueDict=gradient_check_value_dict)
    gradient_check_gradient = gradient_check_ad.bprop(gradient_check_wengert_list,gradient_check_value_dict,loss = np.float_(1.))
    gradient_check_loss_gradient_lhs_all = gradient_check_gradient['W1']

    # compute RHS
    gradient_check_value_dict2 = mlp.my_xman.inputDict()
    gradient_check_wengert_list2 = mlp.my_xman.operationSequence(mlp.my_xman.loss)
    (row,col) = gradient_check_value_dict2['W1'].shape
    ri = np.random.randint(row * col)
    (i,j) = (ri/col,ri % col)

    gradient_check_loss_gradient_lhs = gradient_check_loss_gradient_lhs_all[i,j]

    gradient_check_value_dict2['W1'][i,j] = gradient_check_value_dict2['W1'][i,j] + epsilon

    gradient_check_value_dict2 = gradient_check_ad.eval(gradient_check_wengert_list2,valueDict=gradient_check_value_dict2)

    bigger_loss = gradient_check_value_dict2['loss']

    gradient_check_value_dict3 = mlp.my_xman.inputDict()
    gradient_check_value_dict3['W1'][i,j] = gradient_check_value_dict3['W1'][i,j] - epsilon

    gradient_check_wengert_list3 = mlp.my_xman.operationSequence(mlp.my_xman.loss)
    gradient_check_value_dict3 = gradient_check_ad.eval(gradient_check_wengert_list3,valueDict=gradient_check_value_dict3)

    smaller_loss = gradient_check_value_dict3['loss']
    gradient_check_loss_gradient_rhs = (bigger_loss - smaller_loss)/(2 * epsilon)


    diff = abs(gradient_check_loss_gradient_lhs - gradient_check_loss_gradient_rhs)
    print "gradient check diff: " + str(diff)
    #TODO CHECK GRADIENTS HERE


    print "done"

    # train
    print "training..."
    # get default data and params
    value_dict = mlp.my_xman.inputDict()

    lr = init_lr

    ad = Autograd(mlp.my_xman)

    # TODO prepare wengert list
    wengert_list = mlp.my_xman.operationSequence(mlp.my_xman.loss)

    minLossvalue_dict = None
    minValidationLoss = float('inf')

    for i in range(epochs):
        for (idxs,e,l) in mb_train:
            (numRow,M,V) = e.shape
            value_dict['x'] = np.reshape(e,(numRow,M*V))
            value_dict['y'] = l
            value_dict = ad.eval(wengert_list,valueDict=value_dict)
            gradients = ad.bprop(wengert_list,value_dict,loss = np.float_(1.))

            for rname in gradients:
                if mlp.my_xman.isParam(rname):
                    value_dict[rname] = value_dict[rname] - lr * gradients[rname]
            # TODO prepare the input and do a fwd-bckwd pass over it and update the weights
        # validate
        for (idxs,e,l) in mb_valid:
            (numRow,M,V) = e.shape
            value_dict['x'] = np.reshape(e,(numRow,M*V))
            value_dict['y'] = l
            value_dict = ad.eval(wengert_list,value_dict)
            # TODO prepare the input and do a fwd pass over it to compute the loss
            if value_dict['loss'] < minValidationLoss:
                minValidationLoss = value_dict['loss']
                minLossvalue_dict = deepcopy(value_dict)

            # TODO compare current validation loss to minimum validation loss
        # and store params if needed
    print minValidationLoss
    print "done"

    output_prob = []
    for (idxs,e,l) in mb_test:
        (numRow,M,V) = e.shape
        minLossvalue_dict['x'] = np.reshape(e,(numRow,M*V))
        minLossvalue_dict['y'] = l
        minLossvalue_dict = ad.eval(wengert_list,minLossvalue_dict)
        output_prob.append(minLossvalue_dict['outputs'])
    # prepare input and do a fwd pass over it to compute the output probs
        
    #TODO save probabilities on test set
    print "output result for test set: "
    print np.vstack(output_prob)
    # ensure that these are in the same order as the test input
    np.save(output_file, np.vstack(output_prob))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)
