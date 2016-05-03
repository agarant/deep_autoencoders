Everytime cae.py or dae.py is run the result is saved in ./experiment/config_of_experiment. 

The codes are built to run on a gpu, the following page explains how to install everything related to running theano on gpu
http://deeplearning.net/software/theano/tutorial/using_gpu.html

The file my_conda_env.yml is the conda environment I used to run my tests. It definitely contains some unneeded depencies, but everything is in there.

dae.py is deep autoencoder - > Implemented only on Theano 
cae.py is the deep convolutionnal autoencoder -> Implemented with the help of Lasagne

cae.py additionnaly requires Lasagne
 - pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
 - pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

