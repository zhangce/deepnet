---
title: LeNet MNIST Tutorial
description: Train and test "LeNet" on the MNIST handwritten digit data.
category: example
include_in_docs: true
priority: 1
---

# Training LeNet on MNIST

## Prepare Datasets

You will first need to download and convert the data format from the MNIST website. To do this, simply run the following commands:

If it complains that `wget` or `gunzip` are not installed, you need to install them respectively. After running the script there should be two datasets, `mnist_train_lmdb`, and `mnist_test_lmdb`.

## LeNet: the MNIST Classification Model

Before we actually run the training program, let's explain what will happen. We will use the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) network, which is known to work well on digit classification tasks. We will use a slightly different version from the original LeNet implementation, replacing the sigmoid activations with Rectified Linear Unit (ReLU) activations for the neurons.

The design of LeNet contains the essence of CNNs that are still used in larger models such as the ones in ImageNet. In general, it consists of a convolutional layer followed by a pooling layer, another convolution layer followed by a pooling layer, and then two fully connected layers similar to the conventional multilayer perceptrons. We have defined the layers in `$DEEPNET_ROOT/src/mnist/lenet_train_test.prototxt`.

## Define the MNIST Network

This section explains the `lenet_train_test.prototxt` model definition that specifies the LeNet model for MNIST handwritten digit classification. We assume that you are familiar with [Google Protobuf](https://developers.google.com/protocol-buffers/docs/overview), and assume that you have read the protobuf definitions used by deepnet, which can be found at `$DEEPNET_ROOT/src/parser/cnn.proto`.

Specifically, we will write a `cnn::NetParameter` protobuf. We will start by giving the network a name:

    name: "LeNet"

### Writing the Data Layer

Currently, we will read the MNIST data from the lmdb we created earlier in the demo. This is defined by a data layer:

    layers {
      name: "mnist"
      type: DATA
      data_param {
        source: "mnist_train_lmdb"
        backend: LMDB
        batch_size: 64
        scale: 0.00390625
      }
      top: "data"
      top: "label"
    }

Specifically, this layer has name `mnist`, type `data`, and it reads the data from the given lmdb source. We will use a batch size of 64, and scale the incoming pixels so that they are in the range \[0,1\). Why 0.00390625? It is 1 divided by 256. And finally, this layer produces two output, one is the `data` and other is the `label`.

### Writing the Convolution Layer

Let's define the first convolution layer:

    layers {
      name: "conv1"
      type: CONVOLUTION
      blobs_lr: 1.
      blobs_lr: 2.
      convolution_param {
        num_output: 20
        kernelsize: 5
        stride: 1
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "data"
      top: "conv1"
    }

This layer takes the `data` blob (it is provided by the data layer), and produces the `conv1` layer. It produces outputs of 20 channels, with the convolutional kernel size 5 and carried out with stride 1.

The fillers allow us to randomly initialize the value of the weights and bias. For the weight filler, we will use the `xavier` algorithm that automatically determines the scale of initialization based on the number of input and output neurons. For the bias filler, we will simply initialize it as constant, with the default filling value 0.


### Writing the Pooling Layer

Phew. Pooling layers are actually much easier to define:

    layers {
      name: "pool1"
      type: POOLING
      pooling_param {
        kernel_size: 2
        stride: 2
        pool: MAX
      }
      bottom: "conv1"
      top: "pool1"
    }

This says we will perform max pooling with a pool kernel size 2 and a stride of 2 (so no overlapping between neighboring pooling regions).

Similarly, you can write up the second convolution and pooling layers. Check `$DEEPNET_ROOT/src/mnist/lenet_train_test.prototxt` for details.

### Writing the Fully Connected Layer

Writing a fully connected layer is also simple:

    layers {
      name: "ip1"
      type: INNER_PRODUCT
      blobs_lr: 1.
      blobs_lr: 2.
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "pool2"
      top: "ip1"
    }

This defines a fully connected layer with 500 outputs. All other lines look familiar, right?

### Writing the ReLU Layer

A ReLU Layer is also simple:

    layers {
      name: "relu1"
      type: RELU
      bottom: "ip1"
      top: "ip1"
    }

Since ReLU is an element-wise operation, we can do *in-place* operations to save some memory. This is achieved by simply giving the same name to the bottom and top layer. Of course, do NOT use duplicated names for other layer types!

After the ReLU layer, we will write another innerproduct layer:

    layers {
      name: "ip2"
      type: INNER_PRODUCT
      blobs_lr: 1.
      blobs_lr: 2.
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "ip1"
      top: "ip2"
    }

### Writing the Loss Layer

Finally, we will write the loss!

    layers {
      name: "loss"
      type: SOFTMAX_LOSS
      bottom: "ip2"
      bottom: "label"
    }

The `softmax_loss` layer implements both the softmax and the multinomial logistic loss (that saves time and improves numerical stability). It takes two blobs, the first one being the prediction and the second one being the `label` provided by the data layer (remember it?). It does not produce any outputs - all it does is to compute the loss function value, report it when backpropagation starts, and initiates the gradient with respect to `ip2`. This is where all magic starts.


### Additional Notes: Writing Layer Rules

Layer definitions can include rules for whether and when they are included in the network definition, like the one below:

    layers {
      // ...layer definition...
      include: { phase: TRAIN }
    }

This is a rule, which controls layer inclusion in the network, based on current network's state.
You can refer to `$DEEPNET_ROOT/src/parser/cnn.proto` for more information about layer rules and model schema.

In the above example, this layer will be included only in `TRAIN` phase.
If we change `TRAIN` with `TEST`, then this layer will be used only in test phase.
By default, that is without layer rules, a layer is always included in the network.
Thus, `lenet_train_test.prototxt` has two `DATA` layers defined (with different `batch_size`), one for the training phase and one for the testing phase.
Also, there is an `ACCURACY` layer which is included only in `TEST` phase for reporting the model accuracy every 100 iteration, as defined in `lenet_solver.prototxt`.

## Define the MNIST Solver

Check out the comments explaining each line in the prototxt `$DEEPNET_ROOT/src/parser/mnist/lenet_solver.prototxt`:

    # The train/test net protocol buffer definition
    net: "mnist/lenet_train_test.prototxt"
    # test_iter specifies how many forward passes the test should carry out.
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    test_iter: 100
    # Carry out testing every 500 training iterations.
    test_interval: 500
    # The base learning rate, momentum and the weight decay of the network.
    base_lr: 0.01
    momentum: 0.9
    weight_decay: 0.0005
    # The learning rate policy
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    # Display every 100 iterations
    display: 100
    # The maximum number of iterations
    max_iter: 10000
    # snapshot intermediate results
    snapshot: 5000
    snapshot_prefix: "mnist/lenet"
    # solver mode: CPU or GPU
    solver_mode: GPU


## Training and Testing the Model

MNIST is a small dataset, so training with GPU does not really introduce too much benefit due to communication overheads. On larger datasets with more complex models, such as ImageNet, the computation speed difference will be more significant.
