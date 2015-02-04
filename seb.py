import os
import pylearn2
from pylearn2.config import yaml_parse
path = 'mlp_tutorial_part_2.yaml'
#with open(path, 'r') as f:
train = open(path, 'r')
train = train.read()
hyper_params = {'train_stop' : 50000,
                'valid_stop' : 60000,
                'dim_h0' : 500,
                'max_epochs' : 10000,
                'save_path' : '.'}
train = train % (hyper_params)
print train
train = yaml_parse.load(train)
train.main_loop()
