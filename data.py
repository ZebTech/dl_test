# -*- coding: utf-8 -*-

import os
import inspect
import cPickle as pickle
import numpy as np

from os.path import exists
from skimage.io import imread
from skimage.transform import resize, rotate, swirl
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from random import randint
from math import sqrt

TRAIN_PERCENT = 0.7
VALID_PERCENT = 0.1
TEST_PERCENT = 0.2
SAVE = True


def print_images(images, folder='cropped/'):
    import imageio as imio
    width = sqrt(images.shape[1])
    for i, img in enumerate(images):
        img = img.reshape(width, width)
        imio.imsave(folder + str(i) + '.jpg', img)


class RotationalDDM(DenseDesignMatrix):

    def __init__(self, X, y, y_labels=None):
        self.original_X = X
        super(RotationalDDM, self).__init__(X=X, y=y, y_labels=y_labels)

    def rotation(self, x):
        width = sqrt(x.shape[0])
        angle = randint(0, 359)
        img = x.reshape(width, width)
        return rotate(img, angle, mode='nearest').ravel()

    def parallel_rotate(self, X):
        return [self.rotation(x) for x in X]

    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False):
        self.X = self.parallel_rotate(self.original_X)
        self.X = np.array(self.X)
        print 'Rotated'
        return super(RotationalDDM, self).iterator(
            mode=mode,
            batch_size=batch_size,
            num_batches=num_batches,
            rng=rng,
            data_specs=data_specs,
            return_tuple=return_tuple
        )


class Data(object):

    def __init__(self, size=28, train_perc=TRAIN_PERCENT,
                 valid_perc=VALID_PERCENT, test_perc=TEST_PERCENT,
                 augmentation=0, dataset=None, shuffle=True):
        print 'loading data'
        self.size = size
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.test_perc = test_perc
        self.augmentation = augmentation
        if dataset is None:
            data, targets = self.get_data(size)
        else:
            data, targets = dataset.data, dataset.target
        nb_train = int(self.train_perc * len(targets))
        nb_valid = int(self.valid_perc * len(targets))
        nb_test = int(self.test_perc * len(targets))
        total = nb_train + nb_valid + nb_test
        total_perc = self.train_perc + self.valid_perc + self.test_perc
        data = np.around(data, 4)
        if shuffle:
            data, targets = self.shuffle_data(data[:total], targets[:total])
        self.train_X = data[:nb_train]
        self.train_Y = targets[:nb_train]
        self.valid_X = data[nb_train:nb_train + nb_valid]
        self.valid_Y = targets[nb_train:nb_train + nb_valid]
        self.test_X = data[nb_train + nb_valid:total]
        self.test_Y = targets[nb_train + nb_valid:total]
        name = 'train' + str(size) + '_' + str(total_perc)
        self.save_set(name, data[:total], targets[:total])

    def save_set(self, name, X, y,  directory=''):
        if SAVE:
            curr_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
            filename = os.path.join(curr_dir, directory + name + '.pkl')
            f = open(filename, 'wb')
            pickle.dump((X, y), f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    def convertBinaryValues(self, image_set=None, threshold=0.5):
        binary = np.array(image_set) > threshold
        return binary.astype(int)

    def augment_data(self, image, target):
        images = [image.ravel(), ]
        targets = [target, ]
        image_modifiers = (
            lambda x: rotate(x, 90),
            lambda x: rotate(x, 180),
            lambda x: rotate(x, 270),
            lambda x: rotate(x, 45),
            lambda x: swirl(x)
        )
        for i in xrange(self.augmentation):
            img = image_modifiers[i](image)
            images.append(img.ravel())
            targets.append(target)
        return images, targets

    def create_thumbnail(self, size, img=None):
        print 'processing raw images'
        if img:
            return resize(img, (size, size))
        curr_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        folders = os.walk(os.path.join(curr_dir, '../../data/train/'))
        images = []
        classes = []
        targets = []
        for class_id, folder in enumerate(folders):
            classes.append(folder[0][17:])
            for img in folder[2]:
                if img.index('.jpg') == -1:
                    continue
                image = imread(folder[0] + '/' + img)
                image = resize(image, (size, size))
                # Important to put -1, to have it 0-based.
                target = class_id - 1
                new_images, new_targets = self.augment_data(image, target)
                images.extend(new_images)
                targets.extend(new_targets)
        train = (images, targets)
        self.save_set('train' + str(size), images, targets)
        # f = open(curr_dir + '/train' + str(size) + '.pkl', 'wb')
        # pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
        # f.close()
        return train

    def shuffle_data(self, X, y):
        print 'Shuffling the data...'
        shp = np.shape(X)
        shuffle = np.zeros((shp[0], shp[1] + 1))
        shuffle[:, :-1] = X
        shuffle[:, -1] = y
        np.random.shuffle(shuffle)
        return (shuffle[:, :-1], shuffle[:, -1])

    def get_data(self, size):
        curr_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        filename = os.path.join(curr_dir, 'train' + str(size) + '.pkl')
        total = self.train_perc + self.valid_perc + self.test_perc
        previous_file = os.path.join(
            curr_dir, 'train' + str(size) + '_' + str(total) + '.pkl')
        if exists(previous_file):
            print 'loaded from smaller dump'
            f = open(previous_file, 'rb')
            content = pickle.load(f)
            f.close()
            return content
        if not exists(filename):
            return self.create_thumbnail(size)
        return pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    import time
    start = time.time()
    d = Data(size=28, train_perc=0.1, valid_perc=0.0,
             test_perc=0.1, augmentation=4)
    end = time.time()
    print 'Augmented:' + str(end - start)
    print np.shape(d.train_X)
    start = time.time()
    d = Data(size=28, train_perc=0.1, valid_perc=0.0,
             test_perc=0.1, augmentation=0)
    end = time.time()
    print 'Not Augmented:' + str(end - start)
    print np.shape(d.train_X)
