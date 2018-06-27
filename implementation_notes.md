<!-- Added by: xiny, at:  -->

<!--te-->

# Implementation
Its implementation is built on [ResNet](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet) of [tensorpack](https://github.com/ppwwyyxx/tensorpack). The idea is simple. We only add a few functions and make necessary changes on the original [ResNet](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet). Highlights include:

## training
- EpsilonResnetBase.py

	Implementation of sparsity promoting function with 4 ReLUs and side supervision at the intermediate of the network.
 
- LearningRateSetter.py

	A callback class for the implementation of adaptive learning rate. When the number of discarded layers increases, the adaptive learning rate is activated.
	
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
	
		At the beginning of training, we follow the standard learning rate policy. We start with a learning rate of 0.1 and decrease it by a factor of 10 at epochs 82 and 123. That is lr=0.1 at epoch 1, lr = 0.01 at epoch 82, lr = 0.001 at epoch 123. If the network starts losing layer, the standard learning rate policy will stop and the adaptive learning rate policy begin to work: every time a layer is lost, the learning rate will be set to 0.1 again. For example, if a layer is lost at epoch N, the learning rate will become 0.1 at epoch N+1, will be 0.01 at epoch N+41, and be 0.001 at epoch N+61. 
		
## Testing

We use a variable is\_discarded to show the result of the promoting function S(F(x)) in each step. The standard learning rate policy is applied on ImageNet. Some blocks may have no sufficient epochs to decay to zeros.

We maintain this variable with [tf.train.ExponentialMovingAverage](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage) in our experiments to know the value history in previous steps. Its value of 1 indicates the block is discarded. Before a block decays to zeros, its moving average value may be in the range (0,1) as observed in log.log of ImageNet experiments. Finally, we only prune the blocks whose weights decay to zeros. 
