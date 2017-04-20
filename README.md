# DCGANs-Tensorflow
Implementation of regular GANs and DCGANs in Tensorflow.

Requirements
* Python 2.7
* [Tensorflow v1.0](https://www.tensorflow.org/)

This repo contains code for the [original GANs paper](https://arxiv.org/pdf/1406.2661.pdf),
as well as for [DCGANs](https://arxiv.org/pdf/1511.06434.pdf). A few changes from the original
GANs paper are made, such as the use of lrelu activation functions in the discriminator (as
opposed to maxout activations used in the paper), and the final layer in the generator uses
a tanh activation instead of a sigmoid activation to match the range z is drawn from, [-1, 1].

### Results

Tests were done using MNIST.

Results using regular GANs after about 65 epochs.
![gan](http://i.imgur.com/5m5AyrJ.png)

Results of DCGANs after only 4 epochs.
![dcgan](http://i.imgur.com/dkuKVCp.png)

### Training
To train on your own, simply run `python gan.py` or `python dcgan.py`. To generate images shown
above, run `python createPhotos.py checkpoints/gan/ gan`, or `python createPhotos.py
checkpoints/dcgan/ dcgan`


