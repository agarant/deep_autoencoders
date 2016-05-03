import os
from urllib import urlretrieve

import lasagne
import numpy as np
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import gzip
import tarfile


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        data_x = data_x.reshape(-1, 1, 28, 28)
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)


def load_cifar10():
    xs = []
    ys = []

    if not os.path.exists('cifar-10-batches-py'):
        download('cifar-10-python.tar.gz', 'https://www.cs.toronto.edu/~kriz/')
        tar = tarfile.open("cifar-10-python.tar.gz")
        tar.extractall()
        tar.close()

    for j in range(5):
        d = unpickle('cifar-10-batches-py/data_batch_' + `j + 1`)
        x = d['data']
        y = d['labels']
        xs.append(x)
        ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs) / np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000], axis=0)
    # pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    x_train = _grayscale(x[0:50000, :, :, :])
    y_train = y[0:50000]

    x_train, x_val = x_train[-10000:], x_train[10000:]
    y_train, y_val = y_train[-10000:], y_train[10000:]
    print(len(x_train))
    print(len(x_val))
    x_test = _grayscale(x[50000:, :, :, :])
    y_test = y[50000:]

    return [(theano.shared(np.asarray(x_train, dtype=theano.config.floatX), borrow=True),
             T.cast(theano.shared(np.asarray(y_train, dtype=theano.config.floatX), borrow=True), 'int32')),
            (theano.shared(np.asarray(x_val, dtype=theano.config.floatX), borrow=True),
             T.cast(theano.shared(np.asarray(y_val, dtype=theano.config.floatX), borrow=True), 'int32')),
            (theano.shared(np.asarray(x_test, dtype=theano.config.floatX), borrow=True),
             T.cast(theano.shared(np.asarray(y_test, dtype=theano.config.floatX), borrow=True), 'int32'))]


def _grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], 1024)