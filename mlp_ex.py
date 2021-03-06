# -*- coding: utf-8 -*-

import warnings
import theano
import pylearn2
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score, classification_report

from data import Data
from pdb import set_trace as debug

from pylearn2.space import Conv2DSpace
from pylearn2 import termination_criteria
from pylearn2.models import mlp
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.training_algorithms import bgd, sgd
from pylearn2.costs.mlp import dropout

warnings.filterwarnings("ignore")


def convert_one_hot(data):
    return np.array([[1 if y == c else 0 for c in xrange(10)] for y in data])


def convert_categorical(data):
    return np.argmax(data, axis=1)


def train(d):
    print 'Creating dataset'
    # load mnist here
    # X = d.train_X
    # y = d.train_Y
    # test_X = d.test_X
    # test_Y = d.test_Y
    # nb_classes = len(np.unique(y))
    # train_y = convert_one_hot(y)
    # train_set = DenseDesignMatrix(X=X, y=y)
    train = DenseDesignMatrix(X=d.train_X, y=convert_one_hot(d.train_Y))
    valid = DenseDesignMatrix(X=d.valid_X, y=convert_one_hot(d.valid_Y))
    test = DenseDesignMatrix(X=d.test_X, y=convert_one_hot(d.test_Y))

    print 'Setting up'
    batch_size = 1000
    conv = mlp.ConvRectifiedLinear(
        layer_name='c0',
        output_channels=20,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        # W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    mout = MaxoutConvC01B(
        layer_name='m0',
        num_pieces=4,
        num_channels=96,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    mout2 = MaxoutConvC01B(
        layer_name='m1',
        num_pieces=4,
        num_channels=96,
        irange=.05,
        kernel_shape=[5, 5],
        pool_shape=[4, 4],
        pool_stride=[2, 2],
        W_lr_scale=0.25,
        max_kernel_norm=1.9365
    )
    sigmoid = mlp.Sigmoid(
        layer_name='Sigmoid',
        dim=500,
        sparse_init=15,
    )
    smax = mlp.Softmax(
        layer_name='y',
        n_classes=10,
        irange=0.
    )
    in_space = Conv2DSpace(
        shape=[28, 28],
        num_channels=1,
        axes=['c', 0, 1, 'b']
    )
    net = mlp.MLP(
        layers=[mout, mout2, smax],
        input_space=in_space,
        # nvis=784,
    )
    trainer = bgd.BGD(
        batch_size=batch_size,
        line_search_mode='exhaustive',
        conjugate=1,
        updates_per_batch=10,
        monitoring_dataset={
            'train': train,
            'valid': valid,
            'test': test
        },
        termination_criterion=termination_criteria.MonitorBased(
            channel_name='valid_y_misclass')
    )
    trainer = sgd.SGD(
        learning_rate=0.15,
        cost=dropout.Dropout(),
        batch_size=batch_size,
        monitoring_dataset={
            'train': train,
            'valid': valid,
            'test': test
        },
        termination_criterion=termination_criteria.MonitorBased(
            channel_name='valid_y_misclass')
    )
    trainer.setup(net, train)
    epoch = 0
    while True:
        print 'Training...', epoch
        trainer.train(dataset=train)
        net.monitor()
        epoch += 1

"""
    TODO: Get above .98 with momentum and maxout and dropout. And then add several ones.
"""

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    # debug()
    mnist.data = (mnist.data.astype(float) / 255)
    d = Data(dataset=mnist, train_perc=0.65, valid_perc=0.2, test_perc=0.15,
             shuffle=True)
    train(d=d)
    # train()
