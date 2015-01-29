# -*- coding: utf-8 -*-

import warnings
import theano
import pylearn2
import numpy as np
import cPickle as pk

from data import Data, RotationalDDM

from sklearn.datasets import fetch_mldata
from sklearn import neighbors
from sklearn.metrics import accuracy_score

from pdb import set_trace as debug

from theano import tensor as T

from pylearn2.models import mlp, maxout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import dropout, WeightDecay

warnings.filterwarnings("ignore")


def train(d=None):
    train_X = np.array(d.train_X)
    train_y = np.array(d.train_Y)
    test_X = np.array(d.test_X)
    test_y = np.array(d.test_Y)
    train_y = np.array([[1 if y == c else 0 for c in xrange(
        np.unique(d.train_Y).shape[0])] for y in train_y])
    # train_set = RotationalDDM(
    #     X=train_X, y=train_y, y_labels=np.unique(d.train_Y).shape[0])
    train_set = DenseDesignMatrix(
        X=train_X, y=train_y, y_labels=np.unique(d.train_Y).shape[0])
    print 'Setting up'
    batch_size = 256
    c0 = mlp.ConvRectifiedLinear(
        layer_name='c0',
        output_channels=96,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[4, 4],
        W_lr_scale=0.25,
        # max_kernel_norm=1.9365
    )
    c1 = mlp.ConvRectifiedLinear(
        layer_name='c1',
        output_channels=128,
        irange=.05,
        kernel_shape=[3, 3],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        W_lr_scale=0.25,
        # max_kernel_norm=1.9365
    )
    c2 = mlp.ConvRectifiedLinear(
        layer_name='c2',
        output_channels=128,
        irange=.05,
        kernel_shape=[2, 2],
        pool_shape=[2, 2],
        pool_stride=[2, 2],
        W_lr_scale=0.25,
        # max_kernel_norm=1.9365
    )
    sp0 = mlp.SoftmaxPool(
        detector_layer_dim=16,
        layer_name='sp0',
        pool_size=4,
        sparse_init=512,
    )
    sp1 = mlp.SoftmaxPool(
        detector_layer_dim=16,
        layer_name='sp1',
        pool_size=4,
        sparse_init=512,
    )
    r0 = mlp.RectifiedLinear(
        layer_name='r0',
        dim=512,
        sparse_init=512,
    )
    r1 = mlp.RectifiedLinear(
        layer_name='r1',
        dim=512,
        sparse_init=512,
    )
    out = mlp.Softmax(
        n_classes=np.unique(d.train_Y).shape[0],
        layer_name='output',
        irange=.235,
    )
    epochs = EpochCounter(100)
    layers = [c0, c1, out]
    decay_coeffs = [0.002, 0.002, 1.5]
    in_space = Conv2DSpace(
        shape=[d.size, d.size],
        num_channels=1,
        # axes=['c', 0, 1, 'b'],
    )
    vec_space = VectorSpace(d.size ** 2)
    nn = mlp.MLP(layers=layers, input_space=in_space, batch_size=batch_size)
    trainer = sgd.SGD(
        learning_rate=1e-7,
        cost=SumOfCosts(costs=[
            dropout.Dropout(),
            WeightDecay(decay_coeffs),
        ]),
        batch_size=batch_size,
        train_iteration_mode='even_shuffled_sequential',
        termination_criterion=epochs,
        learning_rule=learning_rule.Momentum(init_momentum=0.9),
    )
    trainer.setup(nn, train_set)
    print 'Learning'
    test_X = vec_space.np_format_as(test_X, nn.get_input_space())
    train_X = vec_space.np_format_as(train_X, nn.get_input_space())
    i = 0
    X = nn.get_input_space().make_theano_batch()
    Y = nn.fprop(X)
    predict = theano.function([X], Y)
    best = 40
    best_iter = -1
    while trainer.continue_learning(nn):
        print '--------------'
        print 'Training Epoch ' + str(i)
        trainer.train(dataset=train_set)
        print 'Evaluating...'
        predictions = np.array(predict(train_X[:2000]))
#        predictions = np.array([predict([f, ])[0] for f in train_X[:2000]])
        print np.min(predictions), np.max(predictions)
        print 'Logloss on train: ' + str(accuracy_score(train_y, predictions))
        # predictions = [predict([f, ])[0] for f in test_X]
        predictions = np.array(predict(test_X))
        print np.min(predictions), np.max(predictions)
        score = accuracy_score(test_y, predictions)
        print 'Logloss on test: ' + str(score)
        best, best_iter = (best, best_iter) if best < score else (score, i)
        print 'Current best: ' + str(best) + ' at iter ' + str(best_iter)
        i += 1
        print ' '

if __name__ == '__main__':
    # d = Data(size=32, train_perc=0.1, test_perc=0.015, valid_perc=0.1, augmentation=0)
    mnist = fetch_mldata('MNIST original')
    d = Data(dataset=mnist, train_perc=0.9, valid_perc=0.0, test_perc=0.1,
             shuffle=False)
    train(d=d)
