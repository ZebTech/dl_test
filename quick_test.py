import cPickle as pk
from plot import plotLines
import numpy as np
import theano
from pylearn2.space import Conv2DSpace, VectorSpace
from pdb import set_trace as debug

mon = np.array(pk.load(open('monitor.pkle')))
axis = [x for x in xrange(len(mon))]
plotLines([[axis, mon[:, 1]], [axis, mon[:, 0]], ], title='Training Graph')

debug()
net = pk.load(open('best.pkle'))
debug()
vec_space = VectorSpace(28 ** 2)
# test_X = vec_space.np_format_as(test_X, net.get_input_space())
# train_X = vec_space.np_format_as(train_X, net.get_input_space())
i = 0
X = net.get_input_space().make_theano_batch()
Y = net.fprop(X)
predict = theano.function([X], Y)
debug()
