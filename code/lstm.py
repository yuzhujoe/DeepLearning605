"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
from copy import deepcopy
import cPickle as cp
from math import sqrt
import time

import numpy as np
from xman import *
from utils import *
from autograd import *

epsilon = 10e-4

np.random.seed(0)


class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.max_len = max_len
        self.in_size = in_size
        self.num_hid = num_hid
        self.out_size = out_size
        self.my_xman= self._build() #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        #TODO: define your model here
        wscale = sqrt(6.0/(self.in_size + self.out_size))
        uscale = sqrt(6.0/(self.out_size + self.out_size))

        # TODO reverse
        input_x_list = []
        for i in xrange(self.max_len):
            input_x_list.append(f.input(name='input_x_'+str(i), default = np.random.rand(1,self.in_size)))

        x.h = f.input(name='h', default = np.zeros((1,self.out_size)))
        x.c = f.input(name='c', default = np.zeros((1,self.out_size)))

        # onehotvector = np.zeros((1,self.out_size))
        # onehotvector[0,0] = 1
        # x.y = f.input(name='y', default = onehotvector)

        x.y = f.input(name='y', default = np.ones((1,self.out_size))/self.out_size)
        x.W2 = f.param(name='W2', default = np.random.uniform(-1*uscale,uscale,(self.out_size,self.out_size)))
        x.b2 = f.param(name='b2', default = np.random.uniform(0,0.1,(self.out_size,)))

        x.Wi = f.param(name='Wi', default = np.random.uniform(-1*wscale,wscale,(self.in_size,self.out_size)))
        x.bi = f.param(name='bi', default = np.random.uniform(0,0.1, (self.out_size,)))
        x.Ui = f.param(name='Ui', default = np.random.uniform(-1*uscale,uscale,(self.out_size,self.out_size)))

        x.Wf = f.param(name='Wf', default = np.random.uniform(-1*wscale,wscale,(self.in_size,self.out_size)))
        x.bf = f.param(name='bf', default = np.random.uniform(0,0.1, (self.out_size,)))
        x.Uf = f.param(name='Uf', default = np.random.uniform(-1*uscale,uscale,(self.out_size,self.out_size)))

        x.Wo = f.param(name='Wo', default = np.random.uniform(-1*wscale,wscale,(self.in_size,self.out_size)))
        x.bo = f.param(name='bo', default = np.random.uniform(0,0.1, (self.out_size,)))
        x.Uo = f.param(name='Uo', default = np.random.uniform(-1*uscale,uscale,(self.out_size,self.out_size)))

        x.Wc = f.param(name='Wc', default = np.random.uniform(-1*wscale,wscale,(self.in_size,self.out_size)))
        x.bc = f.param(name='bc', default = np.random.uniform(0,0.1, (self.out_size,)))
        x.Uc = f.param(name='Uc', default = np.random.uniform(-1*uscale,uscale,(self.out_size,self.out_size)))

        x.temph = x.h
        x.tempc = x.c

        for i in xrange(self.max_len):
            ithreg = input_x_list[self.max_len-1-i]
            x.It = f.sigmoid(f.add(f.add(f.mul(ithreg,x.Wi),f.mul(x.temph,x.Ui)),x.bi))
            x.Ft = f.sigmoid(f.add(f.add(f.mul(ithreg,x.Wf),f.mul(x.temph,x.Uf)),x.bf))
            x.Ot = f.sigmoid(f.add(f.add(f.mul(ithreg,x.Wo),f.mul(x.temph,x.Uo)),x.bo))
            x.C_telta_t = f.tanh(f.add(f.add(f.mul(ithreg,x.Wc),f.mul(x.temph,x.Uc)),x.bc))
            x.c2 = f.add(f.elementMul(x.Ft,x.tempc),f.elementMul(x.It,x.C_telta_t))
            # x.c2.name = "c2_" + str(i)
            x.h2 = f.elementMul(x.Ot,f.tanh(x.c2))
            # x.h2.name = "h2_" + str(i)
            x.temph = x.h2
            x.tempc = x.c2

        x.o2 = f.relu(f.add(f.mul(x.temph,x.W2),x.b2))
        x.outputs = f.softMax(x.o2)
        x.loss = f.mean(f.crossEnt(x.outputs,x.y))

        return x.setup()

def chekgradient(max_len,num_chars,num_hid,num_labels,param):
    lstm = LSTM(max_len,num_chars,num_hid,num_labels)
    # compute LHS
    gradient_check_value_dict = lstm.my_xman.inputDict()
    gradient_check_wengert_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    gradient_check_ad = Autograd(lstm.my_xman)
    gradient_check_value_dict = gradient_check_ad.eval(gradient_check_wengert_list,valueDict=gradient_check_value_dict)
    gradient_check_gradient = gradient_check_ad.bprop(gradient_check_wengert_list,gradient_check_value_dict,loss = np.float_(1.))
    gradient_check_loss_gradient_lhs_all = gradient_check_gradient[param]
    # compute RHS
    gradient_check_value_dict2 = lstm.my_xman.inputDict()
    gradient_check_wengert_list2 = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    (row,col) = gradient_check_value_dict2[param].shape
    ri = np.random.randint(row * col)
    (i,j) = (ri/col,ri % col)

    gradient_check_loss_gradient_lhs = gradient_check_loss_gradient_lhs_all[i,j]
    gradient_check_value_dict2[param][i,j] = gradient_check_value_dict2[param][i,j] + epsilon

    gradient_check_value_dict2 = gradient_check_ad.eval(gradient_check_wengert_list2,valueDict=gradient_check_value_dict2)

    bigger_loss = gradient_check_value_dict2['loss']

    gradient_check_value_dict3 = lstm.my_xman.inputDict()
    gradient_check_value_dict3[param][i,j] = gradient_check_value_dict3[param][i,j] - epsilon

    gradient_check_wengert_list3 = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    gradient_check_value_dict3 = gradient_check_ad.eval(gradient_check_wengert_list3,valueDict=gradient_check_value_dict3)

    smaller_loss = gradient_check_value_dict3['loss']
    gradient_check_loss_gradient_rhs = (bigger_loss - smaller_loss)/(2 * epsilon)

    diff = abs(gradient_check_loss_gradient_lhs - gradient_check_loss_gradient_rhs)
    print diff
    print "gradient check {0} diff: {0:.6s}".format(param,str(diff))


def Accuracy(predict,label):
    (row,col) = predict.shape

    count = 0
    for i in xrange(row):
        if np.argmax(predict[i,:]) == np.argmax(label[i,:]):
            count+=1

    return count*1.0/row

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

    validationSize = len(data.validation)
    testSize = len(data.test)

    # build
    print "building lstm..."

    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)

    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Wi')
    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Ui')
    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Wf')
    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Uf')
    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Wo')
    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Uo')
    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Wc')
    # chekgradient(max_len,mb_train.num_chars,num_hid,mb_train.num_labels,'Uc')

    #TODO CHECK GRADIENTS HERE

    # train
    print "training..."
    # get default data and params
    value_dict = lstm.my_xman.inputDict()
    # lr = init_lr

    lr = 1.5

    ad = Autograd(lstm.my_xman)

    wengert_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)

    minLossvalue_dict = None
    minValidationLoss = float('inf')

    startT = time.time()

    for i in range(epochs):
        for (idxs,e,l) in mb_train:
            value_dict['h'] = np.zeros((e.shape[0],mb_train.num_labels))
            value_dict['c'] = np.zeros((e.shape[0],mb_train.num_labels))
            for j in xrange(max_len):
                value_dict['input_x_'+str(j)] = e[:,j,:]
            value_dict['y'] = l
            value_dict = ad.eval(wengert_list,valueDict=value_dict)
            gradients = ad.bprop(wengert_list,value_dict,loss = np.float_(1.))

            for rname in gradients:
                if lstm.my_xman.isParam(rname):
                    value_dict[rname] = value_dict[rname] - lr * gradients[rname]
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
        # validate
        tempMinLoss = float('inf')
        tempminLossvalue_dict = None

        lr_has_decrease = False
        for (idxs,e,l) in mb_valid:
            value_dict['h'] = np.zeros((e.shape[0],mb_train.num_labels))
            value_dict['c'] = np.zeros((e.shape[0],mb_train.num_labels))
            for j in xrange(max_len):
                # TODO!!! data and label should reverse
                value_dict['input_x_'+str(j)] = e[:,j,:]
            value_dict['y'] = l
            value_dict = ad.eval(wengert_list,value_dict)
            #TODO prepare the input and do a fwd pass over it to compute the loss
            if value_dict['loss'] < tempMinLoss:
                tempMinLoss = value_dict['loss']
                tempminLossvalue_dict = value_dict

        if tempMinLoss < minValidationLoss:
            minValidationLoss = tempMinLoss
            minLossvalue_dict = deepcopy(tempminLossvalue_dict)
        #     if lr >= 1.5:
        #         lr /= 1.2
        # else:
        #     if lr > 0.4:
        #         lr/=1.3

        print minValidationLoss

        #TODO compare current validation loss to minimum validation loss and store params if needed
    print minValidationLoss
    print "done, time is : "
    print time.time() - startT

    output_prob = []
    test_out_label = []

    for (idxs,e,l) in mb_test:
        minLossvalue_dict['h'] = np.zeros((e.shape[0],mb_train.num_labels))
        minLossvalue_dict['c'] = np.zeros((e.shape[0],mb_train.num_labels))
        for j in xrange(max_len):
            minLossvalue_dict['input_x_'+str(j)] = e[:,j,:]
        minLossvalue_dict['y'] = l
        test_out_label.append(l)
        minLossvalue_dict = ad.eval(wengert_list,minLossvalue_dict)
        output_prob.append(minLossvalue_dict['outputs'])
        # prepare input and do a fwd pass over it to compute the output probs
    #TODO save probabilities on test set

    print "minLossvalue: "
    print minLossvalue_dict['loss']
    print "output result for test set: "
    print np.vstack(output_prob)

    print Accuracy(np.vstack(output_prob),np.vstack(test_out_label))

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
