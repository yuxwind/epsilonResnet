Table of Contents
=================

   * [ε-ResNet]()
   * [Implementation]()
   * [Experiments]()
      * [ε-ResNet]()
         * [imagenetEpsilonResnet.py]()
         * [cifarEpsilonResnet.py]()
      * [compress model]()
         * [compressModel.py]()
         * [cifarCompressedResnet.py]()
         * [Discussion on compressing ImageNet models]()
      * [Experiment lists]()
   * [Install]()
      * [Citing ε-ResNet]()


# &epsilon;-ResNet
[&epsilon;-ResNet](https://arxiv.org/abs/1804.01661) is a variant of ResNet to automatically discard redundant layers, which produces responses that are smaller than a threshold &epsilon;, with a marginal or no loss in performance.

# Implementation
Its implementation is built on [ResNet](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet) of [tensorpack](https://github.com/ppwwyyxx/tensorpack). The idea is simple. We only add a few functions and make necessary changes on the original [ResNet](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet). Hightlights include:

- EpsilonResnetBase.py

	Implementation of sparsity promting function with 4 ReLUs and side supervision at the intermediate of the network.
 
- LearningRateSetter.py

	A callback class for the impletmentation of adaptive learning rate. When the number of discarded layers increases, the adaptive learning rate is actived.
	
- imagenetEpsilonResnet.py, cifarEpsilonResnet.py, svhnEpsilonResnet.py

	Training on ImageNet, CIFAR-10, CIFAR-100, SVHN datasets. We make no change on data augmentation. Modifications include: 
	
	+ In \_build\_graph(), strict\_identity() function is applied in residual functions. 
	+ In get_config(), a InferenceRunner() instance is added for side supervision; a LearningRateSetter() instance is added for adaptive learning rate.
	+ The variable discarded_cnt is to count the number of discarded layers.

- Notes on sparse promoting function:

	+ The output of one residual block F(x) is a 4D matrix, that is batch_size x height x width x channel. Only if all of the elements in F(x) is smaller than epsilon, we will have S(F(X))=0. It requires the responses of all the images in one batch smaller than epsilon. 
Thus, the result of S(F(X)) of a residual block is relatively stable because of considering all images in one batch.

	+ As mentioned above, S(F(X)) may be dynamic for different batches. When one batch makes S(F(x)) become 0, the l2 norm in the loss function will begin to force the weights of F(x) to decrease and finally all weights become zero.
	
- Notes on adaptive learning rate
	+ The adaptive learning rate is required to train epsilon-ResNet on Cifar10, cifar100, svhn datasets. The standard learning rate policy will lead to bad performance. 
	+ ImageNet experiments apply the standard learning rate policy.
	+ Let's take the adaptive learning rate on Cifar10 and Cifar100 as example:
	
		At the begging of training, we follow the standard learning rate policy. We start with a learning rate of 0.1 and decrease it by a factor of 10 at epochs 82 and 123. That is lr=0.1 at epoch 1, lr = 0.01 at epoch 82, lr = 0.001 at epoch 123. If the network starts losing layer, the standard learning rate policy will stop and the adaptive learning rate policy begin to work: every time a layer is lost, the learning rate will be set to 0.1 again. For example, if a layer is lost at epoch N, the learning rate will become 0.1 at epoch N+1, will be 0.01 at epoch N+41, and be 0.001 at epoch N+61. 
	 
# Experiments	
## &epsilon;-ResNet
### imagenetEpsilonResnet.py
This is the training code of [&epsilon;-ResNet](https://arxiv.org/abs/1804.01661) on ImageNet. The experiment results on Pre-activatation ResNet(the standard one) and &epsilon;-ResNet of 101 layers are as below. Two &epsilon; values 2.0 and 2.1 give out 20.12% and 25.60% compression ratio seperately.

<p style="text-align:center;"><img src="figures/imagenet-val-error.png" align="middle" width="450" height="300"/></p>

Usage:

```
python imagenetEpsilonResnet.py -d 101 -e 2.0 --gpu 0,1,2,3 --data {path_to_ilsvrc12_data}  
```


### cifarEpsilonResnet.py
It is to train our model on CIFAR-10 and CIFAR-100. The experiment results on Pre-activation ResNet(the orange line), Pre-activation ResNet(the purple line), and &epsilon;-ResNet(the blue line) of 110 layers with &epsilon; of 2.5 are shown as below:

![cifar10-val-error](figures/cifar10-val-error.png)

The following figure shows the adaptive learning rate of this experiment. The two baselines adopt the same learning rate policy. 

![cifar10-lr](figures/cifar10-lr.png)
Usage:

```
python cifarEpsilonResnet.py -n 18 -e 2.5 --gpu 1 -o cifar10-e_2.5-n_18 
```

## compress model
### compressModel.py
The script compressModel.py will compress a model obtained during train. The parameter '--dir' specifies the train_log file directory and '--step' specifies the model of which step is to be compressed.

Usage:

```
python compressModel.py --dir models/cifar10-e_2.5-n_125 --step 303420

```

NOTE: the following files are required in '--dir' 

+ log.log is the log file of training &epsilon;-ResNet
+ model-303420.data-00000-of-00001 and model-303420.index 

It will generate compressed model files:

+ The structure of comopressed model is stored in compressed\_model\_303420.cfg
+ The compressed model: compressed\_model\_303420.data-00000-of-00001, compressed_model_303420.index

### cifarCompressedResnet.py
The script cifarCompressedResnet.py builds a standard ResNet based on the structure information file. It will do inference on the compressed model.

Usage:

``` 
python cifarCompressedResnet.py --cfg models/cifar10-n_125/compressed_model_303420.cfg --gpu 0 --cifar10 
```

A few trained and compressed models on CIFAR-10 and ImageNet are listed [here](https://drive.google.com/drive/folders/1pJ6C3IbxmrvwjgTlnlQ13bLZXuH3j8yt?usp=sharing)

### Discussion on compressing ImageNet models

We use a variable is\_discarded to show the result of the promoting function S(F(x)) in each step. The standard learning rate policy is applied on ImageNet. Some blocks may have no sufficient epochs to decay to zeros.

We maintain this variable with [tf.train.ExponentialMovingAverage](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage) in our experiments to know the value history in previous steps. Its value of 1 indicates the block is discarded. Before a block decays to zeros, its moving average value may be in the range (0,1) as observed in log.log of ImageNet experiments. Finally, we only prune the blocks whose weights decay to zeros. 

<!---
Or we prune a block if the moving average value is greater than a treshold. That's is discarded\_threshold in compressModel.py.

On a ImangeNet model of &epsilon;-ResNet 101, we test different discarded\_treshold and get results as below. 

[//]: # (|  discarded\_treshold | 0.5   |  1 | 0.3  |)
[//]: # (|---|---|---|---|)
[//]: # (| val\_error\_top1  	|0.23036 |  0.23182 |  0.23448  | )
[//]: # (| val\_error\_top5  	|0.06694 |  0.06744 |  0.0694   |  )
[//]: # (| #discarded block  |7   	  |  6|  8 |  ) 
-->

## Experiment lists

If you would like to compare with our experiements in your research, please run with parameters in the following scripts directly.

+ run_cifar10.sh
+ run_cifar100.sh
+ run_imagenet.sh
+ run_svhn.sh

# Install

Dependencies is the same as [tensorpack](https://github.com/ppwwyyxx/tensorpack):

+ Python 2.7 or 3
+ Python bindings for OpenCV (Optional)
+ TensorFlow >= 1.3.0

```
# install git, then:
# pull tensorpack
git clone https://github.com/ppwwyyxx/tensorpack.git
# This implementation is based on tags/0.2.0. I noticed that some APIs in the latest tensorpack is changed, I'll make it compatible in the future version.  
git checkout tags/0.2.0

# pull epsilon-ResNet
git clone https://github.com/yuxwind/epsilon-resnet.git

# put LearningRateSetter.py to {tensorpack_root}/tensorpack/callbacks/
```

## Citing &epsilon;-ResNet

Please cite &epsilon;-ResNet in your publication if it helps your research:

```
@article{DBLP:yu2018EpsilonResNet,
  author    = {Xin Yu and Zhiding Yu and Srikumar Ramalingam},
  title     = {Learning Strict Identity Mappings in Deep Residual Networks},
  journal   = {CoRR},
  volume    = {abs/1804.01661},
  year      = {2018}
}
```
