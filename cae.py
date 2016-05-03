import theano
import lasagne
import os
import random
from datetime import datetime
import numpy as np
import theano.tensor as T
from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, BiasLayer, NonlinearityLayer, \
    GlobalPoolLayer, BatchNormLayer, InverseLayer, get_output, get_all_params
from lasagne.objectives import binary_crossentropy
from lasagne.updates import nesterov_momentum, sgd
from lasagne.nonlinearities import rectify, linear, sigmoid
from datasets_loaders import load_data, load_cifar10

try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils import tile_raster_images
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Cae(object):
    """ Convolutional auto-encoder class (Cae)

    A convolutional autoencoder model is obtained stacking several convolutional layers.
    Convolutions can be followed by a max pooling layer. At the end of the encoding a
    fully connected layer represents the code layer. Tied-weights are used to inverse the
    convolutional layer. However, each layer in the deconvolution part has its own bias
    (or batch-normalization) layer + non linearity.
    Reference papers:
     - http://arxiv.org/abs/1506.02351
     - https://www.cs.toronto.edu/~hinton/science.pdf

    """

    def __init__(
            self,
            input_size,
            layers_config=[(64, 8, 2, 'valid'), (128, 3, 2, 'same')],
            code_layer_size=2,
            batch_norm=True,
            nonlinearity=rectify
    ):
        """"This class is made to support a variable number of layers.

        :type input_size: tuple of int
        :param input_size: Shape of the input i.e (None, 1, 28, 28) Means that it will have a defined at runtime
                           amount of examples with one channel and of size 28 x 28.

        :type layers_config: list of tuples of ints
        :param layers_config: Configuration of the net. i.e. [(64, 5, 2, 'valid'), (32, 3, None, 'same')]
                              Means the first layers will output 64 feature maps, use filters of size
                              of 5 and be followed by a max-pooling layer of with a pool-size of 2.
                              The second layer will output 32 feature maps, use filters of size 3 and
                              will not be followed by a pooling layer. The 4th param is the padding. see:
                              http://lasagne.readthedocs.org/en/latest/modules/layers/conv.html#lasagne.layers.Conv2DLayer

        :type code_layer_size: int
        :param code_layer_size: Determine the size of the code layer.

        :type batch_norm: bool
        :param batch_norm: If True, batch-normalization will be used. Otherwise, bias will be used.

        :type nonlinearity: Lasagne.nonlinearities
        :param nonlinearity: Define the activation function to use
        """

        def bias_plus_nonlinearity(l, bias, nl):
            l = bias(l)
            l = NonlinearityLayer(l, nonlinearity=nl)
            return l

        self.x = T.tensor4('inputs')  # the data is presented as rasterized images

        self.normalization_layer = BatchNormLayer if batch_norm else BiasLayer
        self.nonlinearity = nonlinearity
        self.code_layer_size = code_layer_size
        self.network_config_string = ""

        l = InputLayer(input_var=self.x, shape=input_size)
        invertible_layers = []  # Used to keep track of layers that will be inverted in the decoding phase
        """" Encoding """
        for layer in layers_config:
            l = Conv2DLayer(l, num_filters=layer[0], filter_size=layer[1], nonlinearity=None, b=None,
                            W=lasagne.init.GlorotUniform(), pad=layer[3])
            
            invertible_layers.append(l)
            self.network_config_string += "(" + str(layer[0]) + ")" + str(layer[1]) + "c"
            print(l.output_shape)
            bias_plus_nonlinearity(l, self.normalization_layer, self.nonlinearity)
            if layer[2] is not None:  # then we add a pooling layer
                l = MaxPool2DLayer(l, layer[2])
                invertible_layers.append(l)
                self.network_config_string += "-" + str(layer[2]) + "p"
                print(l.output_shape)
            self.network_config_string += "-"

        # l = DenseLayer(l, num_units=l.output_shape[1], nonlinearity=None, b=None)
        # invertible_layers.append(l)
        # self.network_config_string += str(l.output_shape[1]) + "fc"
        # print(l.output_shape)
        l = DenseLayer(l, num_units=self.code_layer_size, nonlinearity=None, b=None)
        invertible_layers.append(l)
        self.network_config_string += str(self.code_layer_size) + "fc"
        print(l.output_shape)
        # Inspired by Hinton (2006) paper, the code layer is linear which allows to retain more info especially with
        # with code layers of small dimension
        l = bias_plus_nonlinearity(l, self.normalization_layer, linear)
        self.code_layer = get_output(l)

        """ Decoding """
        # l = InverseLayer(l, invertible_layers.pop())  # Inverses the fully connected layer
        # print(l.output_shape)
        # l = bias_plus_nonlinearity(l, self.normalization_layer, self.nonlinearity)
        l = InverseLayer(l, invertible_layers.pop())  # Inverses the fully connected layer
        print(l.output_shape)
        l = bias_plus_nonlinearity(l, self.normalization_layer, self.nonlinearity)
        for i, layer in enumerate(layers_config[::-1]):
            if layer[2] is not None:
                l = InverseLayer(l, invertible_layers.pop())  # Inverse a max-pooling layer
                print(l.output_shape)
            l = InverseLayer(l, invertible_layers.pop())  # Inverse the convolutional layer
            print(l.output_shape)
            # last layer is a sigmoid because its a reconstruction and pixels values are between 0 and 1
            nl = sigmoid if i is len(layers_config) - 1 else self.nonlinearity
            l = bias_plus_nonlinearity(l, self.normalization_layer, nl)  # its own bias_nonlinearity

        self.network = l
        self.reconstruction = get_output(self.network)
        self.params = get_all_params(self.network, trainable=True)
        # Sum on axis 1-2-3 as they represent the image (channels, height, width). This means that we obtain the binary
        # _cross_entropy for every images of the mini-batch which we then take the mean.
        self.fine_tune_cost = T.sum(binary_crossentropy(self.reconstruction, self.x), axis=(1, 2, 3)).mean()
        self.test_cost = T.sum(binary_crossentropy(get_output(self.network), self.x), axis=(1,2,3)).mean()

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

        :type learning_rate: float (usually a shared variable so it can be updated)
        :param learning_rate: learning rate used during finetune stage
        """

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        index = T.lscalar('index')  # index to a [mini]batch

        updates = nesterov_momentum(self.fine_tune_cost, self.params, learning_rate=learning_rate, momentum=0.9)
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
                outputs=self.test_cost,
                givens={
                    self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                },
                name='valid'
        )

        test_score_i = theano.function(
                [index],
                outputs=self.test_cost,
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
        return recons.reshape((n_recon_batches * batch_size, input_shape[-2], input_shape[-1]))

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
        return code_layers.reshape((n_recon_batches * batch_size, code_layers.shape[-1]))


def test_cae(n_epochs=30, batch_size=128, learning_rate=0.01):
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training
    n_train_batches = T.shape(train_set_x)[0].eval() // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print("... building convolutional autoencoder")
    x = train_set_x.get_value(borrow=True)
    cae = Cae(
            input_size=(None, x.shape[1], x.shape[2], x.shape[3])
    )
    # Create the dir that will contain the experiments results
    dir = "./experiments/"+cae.network_config_string + "_" + str(datetime.now())
    os.makedirs(dir)

    # We take 100 random points from the test datasets, we will create a input and then reconstruction of those
    # points to have a visual idea of the quality of the reconstruction
    hundred_indices = sorted(
            random.sample(xrange(test_set_x.get_value(borrow=True).shape[0]),
                          100))  # http://stackoverflow.com/a/6482922
    test_sample = np.array([test_set_x.get_value(borrow=True)[i] for i in hundred_indices])
    image = Image.fromarray(tile_raster_images(X=test_sample, img_shape=(test_sample.shape[2], test_sample.shape[3]),
                                               tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save(dir+"/sample_test_input_images.png")

    print("... building finetuning functions")
    sh_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_model, valid_scorer, test_scorer = cae.build_finetune_functions(datasets, batch_size, sh_lr)

    ###############
    # TRAIN MODEL #
    ###############
    print("... training the model")

    train_costs = []  # Will be use to plot the learning curves
    valid_costs = []
    epoch = 0
    while epoch < n_epochs:
        start_time = time.time()
        train_cost = []
        for minibatch_index in range(n_train_batches):
            train_cost.append(train_model(minibatch_index))
        this_train_cost = np.mean(train_cost)
        train_costs.append(this_train_cost)
        print('Completed epochs %d, training cost %f' % (epoch, this_train_cost))
        epoch += 1  # as the update have passed we now have trained on 1 epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - start_time))

        # Calculate the cost on the validation set
        this_valid_cost = np.mean(valid_scorer())
        valid_costs.append(this_valid_cost)
        print('Completed epochs %d, validation cost: %f' % (epoch, this_valid_cost))

        # Update the learning rate
        if epoch == 10 or epoch == 20 :
            new_lr = sh_lr.get_value() * 0.1
            print("New LR:" + str(new_lr))
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
    test_cost = np.mean(test_scorer())
    print('Final test cost %f' % (test_cost))
    open(dir+'/'+str(test_cost), 'w').close()
    #####################
    # VISUALIZE RESULTS #
    #####################

    # Save an image of a 100 reconstructed characters
    sample_reconstruction = cae.reconstruct(theano.shared(test_sample, borrow=True))
    image = Image.fromarray(
            tile_raster_images(X=sample_reconstruction, img_shape=(test_sample.shape[2], test_sample.shape[3]),
                               tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save(dir+"/sample_test_reconstruction_images.png")

    # Save the learning cost graphs
    l1, l2 = plt.plot(range(n_epochs + 1)[:-1], train_costs, 'r', range(n_epochs + 1)[1:], valid_costs, 'b')
    plt.legend((l1, l2), ('train', 'val'), loc=1)
    plt.savefig(dir+'/learning.png')
    plt.clf()

    # Create a 2d visualisation
    if cae.code_layer_size is not 2:
        return
    x = []
    y = []
    colors = []
    class_colours = ['b', 'g', 'r', 'c', 'peru', 'y', 'lime', 'orange', 'brown', 'deeppink']
    labels = test_set_y.eval()
    for i, code in enumerate(cae.encode(test_set_x)):
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


if __name__ == '__main__':
    test_cae()
