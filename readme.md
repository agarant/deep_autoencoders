
### Implementations of deep autoencoders in Theano/Lasagne.
  - dae.py -> Deep autoencoder (fully connected) pre-trained with stacked denoising autoencoders. (Theano only)
  - cae.py -> Deep convolutionnal autoencoder. (Using Lasagne)

The [Deeplearning tutorial](https://github.com/lisa-lab/DeepLearningTutorials) denoising autoencoder (dA.py) is used to pre-train the deep autoencoder.

The implementations are flexible so that the network configuration can easily be changed. Experiments results are saved in: ./experiment/config_of_experiment. 

### Usage
The autoencoders are built to run on a gpu, the following [page](http://deeplearning.net/software/theano/tutorial/using_gpu.html) explains how to install everything related to running theano on gpu.

The file my_conda_env.yml is the conda environment I used to run my tests. It contains some unneeded dependencies, but everything is in there. To install the environement simply use:
>conda env create -f my_conda_env.yml

## Results

Here are some figures from my experiments. For more details, see: exploration_of_deep_autoencoders_architectures_for_dimensionality_reduction.pdf

![Alt text](/reconstructions.png?raw=true "Reconstructions")

![Alt text](/2d-visualization.png?raw=true "2d Visualisations")