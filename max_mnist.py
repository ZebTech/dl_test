# -*- coding: utf-8 -*-

import warnings
import theano
import pylearn2
import numpy as np

from data import Data, RotationalDDM

from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score, classification_report

from pdb import set_trace as debug

from pylearn2.models import mlp, maxout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.training_algorithms.sgd import LinearDecayOverEpoch
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.costs.cost import SumOfCosts, MethodCost
from pylearn2.costs.mlp import dropout, WeightDecay

warnings.filterwarnings("ignore")


def ac_score(y, pred):
    s = np.array(np.array(y) == np.array(pred))
    return float(np.sum(s)) / float(len(s))


def convert_one_hot(data):
    return np.array([[1 if y == c else 0 for c in xrange(
        len(np.unique(data)))] for y in data])


def convert_categorical(data):
    return np.argmax(data, axis=1)


def train(d=None):
    train_X = np.array(d.train_X)
    train_y = np.array(d.train_Y)
    valid_X = np.array(d.valid_X)
    valid_y = np.array(d.valid_Y)
    test_X = np.array(d.test_X)
    test_y = np.array(d.test_Y)
    nb_classes = len(np.unique(train_y))
    train_y = convert_one_hot(train_y)
    valid_y = convert_one_hot(valid_y)
    # train_set = RotationalDDM(X=train_X, y=train_y)
    train_set = DenseDesignMatrix(X=train_X, y=train_y)
    valid_set = DenseDesignMatrix(X=valid_X, y=valid_y)
    print 'Setting up'
    batch_size = 100
    c0 = mlp.ConvRectifiedLinear(
        layer_name='c0',
        output_channels=64,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        # W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    c1 = mlp.ConvRectifiedLinear(
        layer_name='c1',
        output_channels=64,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        # W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    c2 = mlp.ConvRectifiedLinear(
        layer_name='c2',
        output_channels=64,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[5, 4],
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
    s0 = mlp.Sigmoid(
        layer_name='s0',
        dim=nb_classes,
        max_col_norm=1.9365,
        sparse_init=nb_classes,
    )
    out = mlp.Softmax(
        n_classes=nb_classes,
        layer_name='output',
        # irange=.235,
        max_col_norm=1.9365,
        sparse_init=nb_classes,
    )
    epochs = EpochCounter(100)
    layers = [c0, s0]
    decay_coeffs = [.00005, .00005, .00005]
    in_space = Conv2DSpace(
        shape=[d.size, d.size],
        num_channels=1,
    )
    vec_space = VectorSpace(d.size ** 2)
    nn = mlp.MLP(layers=layers, input_space=in_space, batch_size=batch_size)
    trainer = sgd.SGD(
        learning_rate=0.01,
        cost=SumOfCosts(costs=[
            # dropout.Dropout(),
            # MethodCost(method='cost_from_X'),
            # WeightDecay(decay_coeffs),
        ]),
        batch_size=batch_size,
        # train_iteration_mode='even_shuffled_sequential',
        termination_criterion=epochs,
        learning_rule=learning_rule.Momentum(init_momentum=0.5),
    )
    lr_adjustor = LinearDecayOverEpoch(
        start=1,
        saturate=10,
        decay_factor=.1,
    )
    momentum_adjustor = learning_rule.MomentumAdjustor(
        final_momentum=.99,
        start=1,
        saturate=10,
    )
    trainer.setup(nn, train_set)
    print 'Learning'
    test_X = vec_space.np_format_as(test_X, nn.get_input_space())
    train_X = vec_space.np_format_as(train_X, nn.get_input_space())
    i = 0
    X = nn.get_input_space().make_theano_batch()
    Y = nn.fprop(X)
    predict = theano.function([X], Y)
    best = -40
    best_iter = -1
    while trainer.continue_learning(nn):
        print '--------------'
        print 'Training Epoch ' + str(i)
        trainer.train(dataset=train_set)
        nn.monitor()
        print 'Evaluating...'
        predictions = convert_categorical(predict(train_X[:2000]))
        score = accuracy_score(
            convert_categorical(train_y[:2000]), predictions)
        print 'Score on train: ' + str(score)
        predictions = convert_categorical(predict(test_X))
        score = accuracy_score(test_y, predictions)
        print 'Score on test: ' + str(score)
        best, best_iter = (best, best_iter) if best > score else (score, i)
        print 'Current best: ' + str(best) + ' at iter ' + str(best_iter)
        print classification_report(test_y, predictions)
        print 'Adjusting parameters...'
        # momentum_adjustor.on_monitor(nn, valid_set, trainer)
        # lr_adjustor.on_monitor(nn, valid_set, trainer)
        i += 1
        print ' '

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    mnist.data = np.rint(mnist.data / 255)
    d = Data(dataset=mnist, train_perc=0.8, valid_perc=0.1, test_perc=0.1,
             shuffle=True)
    train(d=d)
