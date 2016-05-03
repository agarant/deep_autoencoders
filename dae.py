"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for Dae.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

from __future__ import print_function

import os
import timeit
from datetime import datetime, time

import numpy as np

import theano
import theano.tensor as T
import random

from mlp import HiddenLayer
from dA import dA

try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils import tile_raster_images
from logistic_sgd import load_data
import time
from datasets_loaders import load_cifar10
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Dae(object):
    """Deep auto-encoder class (Dae)

    A deep autoencoder model is obtained by stacking several fully connected layers
    to form a encoder and a decoder. Stacked denoising auto-encoder are used to pre-train
    the model. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. A deeep-autoencoder tries to reconstruct the input
    as a standard auto-encoder, the exception being that it uses more latent layers
    to encode and then decode the input.
    """

    def __init__(
            self,
            input_size,
            layers_config=[1000, 500, 250, 30],
            nonlinearity=T.nnet.sigmoid
    ):
        """ This class is made to support a variable number of layers.

        :type input_size: tuple of int
        :param input_size: Shape of the input i.e (1, 784) Means that it has 1 channel with
                           size 784.

        :type layers_config: list of ints
        :param layers_config: intermediate layers size, must contain at least one value
        """
        numpy_rng = np.random.RandomState(89677)
        print('start')
        self.x = T.matrix('x')   # the data is presented as rasterized images

        self.hidden_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(layers_config)
        self.network_config_string = ""

        assert self.n_layers > 0

        """ Encoding """
        #  input_layer = self.x.reshape((-1, -1, self.x.shape[1]*self.x.shape[2]))
        for i, layer_size in enumerate(layers_config):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            n_in = layers_config[i - 1] if i > 0 else input_size[1]
            print(str(n_in) + ' , ' + str(layer_size))
            layer_input = self.hidden_layers[-1].output if i > 0 else self.x
            # Inspired by Hinton (2006) paper, the code layer is linear which allows to retain more info especially with
            # with code layers of small dimension
            activation = nonlinearity if i < len(layers_config)-1 else linear
            l = HiddenLayer(rng=numpy_rng,
                            input=layer_input,
                            n_in=n_in,
                            n_out=layer_size,
                            activation=activation)
            self.hidden_layers.append(l)
            self.params.extend(l.params)
            # Construct a denoising autoencoder that shared weights with this layer
            da_layer = dA(numpy_rng=numpy_rng,
                          input=layer_input,
                          n_visible=n_in,
                          n_hidden=layer_size,
                          W=l.W,
                          bhid=l.b)
            self.dA_layers.append(da_layer)
            self.network_config_string += str(layer_size)+"_"

        self.code_layer = self.hidden_layers[-1].output
        """ Decoding """
        for i, layer_size in enumerate(layers_config[::-1]):
            output_size = layers_config[-(i + 2)] if i < len(layers_config) - 1 else input_size[1]
            print(str(layer_size) + ' , ' + str(output_size))
            l = HiddenLayer(rng=numpy_rng,
                            input=self.hidden_layers[-1].output,
                            n_in=layer_size,
                            n_out=output_size,
                            W=self.hidden_layers[len(layers_config) - i - 1].W.T,  # Tied weights
                            activation=T.nnet.sigmoid)
            self.hidden_layers.append(l)
            self.params.extend([l.b])  # since we are using tied weight only the b is a param

        self.reconstruction = self.hidden_layers[-1].output
        a = T.sum(T.nnet.binary_crossentropy(self.reconstruction, self.x), axis=1)
        self.fine_tune_cost = a.mean()


    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        print("Length of dA_layers ", len(self.dA_layers))
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level, learning_rate)
            # compile the theano function
            fn = theano.function(
                    inputs=[
                        index,
                        theano.In(corruption_level, value=0.1),
                        theano.In(learning_rate, value=0.01)
                    ],
                    outputs=cost,
                    updates=updates,
                    givens={
                        self.x: train_set_x[batch_begin: batch_end]
                    }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """Generates a function `train_function` that implements one step of
        finetuning, a function `valid_score` that computes the cost on the validation set
        and a function 'test_score' that computes the reconstruction cost on the test set.

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        """

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        index = T.lscalar('index')  # index to a [mini]batch

        gparams = T.grad(self.fine_tune_cost, self.params)
        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
            ]
        train_function = theano.function(
                inputs=[index],
                outputs=self.fine_tune_cost,
                updates=updates,
                givens={
                    self.x: train_set_x[index * batch_size: (index + 1) * batch_size]
                },
                name='train')

        valid_score_i = theano.function(
                [index],
                outputs=self.fine_tune_cost,
                givens={
                    self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                },
                name='valid'
        )

        test_score_i = theano.function(
                [index],
                outputs=self.fine_tune_cost,
                givens={
                    self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                },
                name='test'
        )

        # Create a function that scans the entire validation set
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_function, valid_score, test_score

    def reconstruct(self, input_set):
        """ Reconstruct the input by passing it through the autoencoder
        """
        index = T.lscalar('index')  # index to a [mini]batch
        input_shape = input_set.get_value(borrow=True).shape
        batch_size = 128 if input_shape[0] > 128 else input_shape[0]  # Set a batch_size to avoid memory problems

        reconstruction = theano.function(
                [index],
                outputs=self.reconstruction,
                givens={
                    self.x: input_set[index * batch_size: (index + 1) * batch_size],
                }
        )
        # Create a function that scans the entire validation set
        n_recon_batches = input_shape[0] // batch_size

        def complete_reconstruction():
            return [reconstruction(i) for i in range(n_recon_batches)]

        recons = np.array(complete_reconstruction())
        return recons.reshape((n_recon_batches*batch_size, input_shape[-1]))

    def encode(self, input_set):
        """ Encode the input by passing it through the autoencoder
        """
        index = T.lscalar('index')  # index to a [mini]batch
        input_shape = input_set.get_value(borrow=True).shape
        batch_size = 128 if input_shape[0] > 128 else input_shape[0]  # Set a batch_size to avoid memory problems

        reconstruction = theano.function(
                [index],
                outputs=self.code_layer,
                givens={
                    self.x: input_set[index * batch_size: (index + 1) * batch_size],
                }
        )
        # Create a function that scans the entire validation set
        input_shape = input_set.get_value(borrow=True).shape
        n_recon_batches = input_shape[0] // batch_size

        def complete_reconstruction():
            return [reconstruction(i) for i in range(n_recon_batches)]

        code_layers = np.array(complete_reconstruction())
        return code_layers.reshape((n_recon_batches*batch_size, code_layers.shape[-1]))


def test_dae(finetune_lr=0.01, pretraining_epochs=20,
             pretrain_lr=0.01, training_epochs=400,
             dataset='mnist.pkl.gz', batch_size=128):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :param training_epochs:
    :param batch_size:
    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data('mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training
    n_train_batches = T.shape(train_set_x)[0].eval() // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print("... building deep autoencoder")
    x = train_set_x.get_value(borrow=True)
    print(x.shape)
    dae = Dae(
            input_size=(1, x.shape[1])
    )
    # Create the dir that will contain the experiments results
    dir = "./experiments/"+dae.network_config_string + "_" + str(datetime.now())
    os.makedirs(dir)

    # We take 100 random points from the test datasets, we will create a input and then reconstruction of those
    # points to have a visual idea of the quality of the reconstruction
    hundred_indices = sorted(
            random.sample(xrange(test_set_x.get_value(borrow=True).shape[0]),
                          100))  # http://stackoverflow.com/a/6482922
    test_sample = np.array([train_set_x.get_value(borrow=True)[i] for i in hundred_indices])
    image = Image.fromarray(tile_raster_images(X=test_sample, img_shape=(28, 28),
                                               tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save(dir+"/sample_test_input_images.png")
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = dae.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)
    print('... pre-training the model')
    corruption_levels = [.4 for i in xrange(10)]

    for i in range(dae.n_layers):
        print("pretraining i: ", i)
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))
    # Save an image of a 100 reconstructed characters
    sample_reconstruction = dae.reconstruct(theano.shared(test_sample, borrow=True))
    image = Image.fromarray(
            tile_raster_images(X=sample_reconstruction, img_shape=(28, 28),
                               tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save(dir+"/sample_test_reconstruction_images_post_pre-training.png")


    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    sh_lr = theano.shared(finetune_lr)
    train_model, valid_scorer, test_scorer = dae.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr
    )
    test_cost = np.mean(test_scorer())
    print('Final test cost %f' % (test_cost))
    open(dir+'/pre_'+str(test_cost), 'w').close()

    print('... finetunning the model')
    train_costs = []  # Will be use to plot the learning curves
    valid_costs = []
    epoch = 0
    while epoch < training_epochs:
        start_time = time.time()
        train_cost = []
        for minibatch_index in range(n_train_batches):
            train_cost.append(train_model(minibatch_index))
        this_train_cost = np.mean(train_cost)
        train_costs.append(this_train_cost)
        print('Completed epochs %d, training cost %f' % (epoch, this_train_cost))
        epoch += 1  # as the update have passed we now have trained on 1 epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, training_epochs, time.time() - start_time))

        # Calculate the cost on the validation set
        this_valid_cost = np.mean(valid_scorer())
        valid_costs.append(this_valid_cost)
        print('Completed epochs %d, validation cost: %f' % (epoch, this_valid_cost))

        # Update the learning rate
        if epoch == 100 or epoch == 200 or epoch == 300:
            new_lr = sh_lr.get_value() * 0.1
            print("New LR:" + str(new_lr))
            sh_lr.set_value(new_lr)
    test_cost = np.mean(test_scorer())
    print('Final test cost %f' % (test_cost))
    open(dir+'/'+str(test_cost), 'w').close()

    #####################
    # VISUALIZE RESULTS #
    #####################
    # Save an image of a 100 reconstructed characters
    sample_reconstruction = dae.reconstruct(theano.shared(test_sample, borrow=True))
    image = Image.fromarray(
            tile_raster_images(X=sample_reconstruction, img_shape=(28, 28),
                               tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save(dir+"/sample_test_reconstruction_images_post_finetuning.png")

    # Save the learning cost graphs
    l1, l2 = plt.plot(range(training_epochs + 1)[:-1], train_costs, 'r', range(training_epochs + 1)[1:], valid_costs, 'b')
    plt.legend((l1, l2), ('train', 'val'), loc=1)
    plt.savefig(dir+'/learning.png')
    plt.clf()

    # Create a 2d visualisation
    # if dae.code_layer.shape[1] is not 2:
    #     return
    x = []
    y = []
    colors = []
    class_colours = ['b', 'g', 'r', 'c', 'peru', 'y', 'lime', 'orange', 'brown', 'deeppink']
    labels = test_set_y.eval()
    for i, code in enumerate(dae.encode(test_set_x)):
        x.append(code[0])
        y.append(code[1])
        colors.append(class_colours[labels[i]])

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    plt.legend(recs, classes, loc=4)
    plt.scatter(x, y, c=colors)
    plt.savefig(dir+'/2d_visualisation.png')


def linear(x):
    return x


if __name__ == '__main__':
    test_dae()
