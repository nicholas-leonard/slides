class: center, middle

# Torch 7: Applied Deep Learning for Vision and Natural Language

Nicholas Leonard

Element Inc.

October 8, 2015

---

# Agenda

1. Introduction - 5 min
2. Packages - 5 min
2. Tensors – 10 min
3. Logistic Regression – 5 min
4. Deep Learning - 2 min
5. Multi-Layer Perceptron- 5 min
5. Convolutional Neural Network – 10 min
6. Recurrent Neural Network – 10 min
7. Hyper-optimization – 5 min

---

# Introduction

My background:

 * 2003-2008 : Bacc. Degree in Computer Science at Royal Military College of Canada ;
 * 2008-2012 : Army Signals Officer : management (office politics, emails), no code, no science ;
 * 2012-2014 : Master's Degree in Deep Learning at University of Montreal ;
  * LISA/MILA lab ;
  * Yoshua Bengio and Aaron Courville as co-directors ;
  * 2012-2013 : Python, Theano and Pylearn2 ;
  * 2013-today : Lua, Torch7 ;
 * 2014-today : employed at Element Inc. :
  * biometrics startup ;
  * deep learning on smart phones (Android, iOS) ;
  * open source contributions (Torch7).
  
---

## Introduction - Lua

Why take the time to learn Lua :

 * easy interface between low-level C/CUDA/C++ and high-level Lua ;
 * light-weight : used for embedded systems ;
 * tables :
   * can be used as lists, dictionaries, classes and objects ;
   * make it easy to extend existing classes (at any level) ;
 * fast for-loops (LuaJIT) ;
 * closures ;

Example : 

```lua
a = {1,2,a=3, print=function(self) print(self) end}
a:print() -- i.e. a.print(a)
```

Output :

```lua
{
  1 : 1
  2 : 2
  print : function: 0x417f11e0
  a : 3
}
```

---

## Introduction - Torch 7

What's up with Torch 7?

  * a powerful N-dimensional array ;
  * lots of routines for indexing, slicing, transposing, ... ;
  * amazing interface to C, via LuaJIT ;
  * linear algebra routines ;
  * easy modular neural networks ;
  * numeric optimization routines ;
  * fast and efficient GPU support ;
  * embeddable, with ports to iOS, Android and FPGA backends ;
  * under development since October 2002 ;
  * used by Facebook, Google [DeepMind], Twitter, NYU, ...


---

## Packages

The Torch 7 distribution is made up of different packages, each its own github repository :

 * __torch7__/__cutorch__ : tensors, BLAS, file I/O (serialization), utilities for unit testing and cmd-line argument parsing ;
 * __nn__/__cunn__ : easy and modular way to build and train simple or complex neural networks using `modules` and `criterions` ;
 * __optim__ : optimization package for nn. Provides training algorithms like SGD, LBFGS, etc. Uses closures ;
 * __trepl__ : torch read–eval–print loop, Lua interpreter, `th>` ;
 * __paths__ : file system manipulation package ;
 * __image__ : for saving, loading, constructing, transforming and displaying images ;

Refer to the torch.ch website for a more complete list of official packages.
 
---

## Packages - Unofficial

Many more unofficial packages out there :

 * __dp__ : deep learning library for cross-validation (early-stopping). An alternative to optim inspired by Pylearn2. Lots of documentation and examples ;
 * __dpnn__ : extensions to the nn library. REINFORCE algorithm ;
 * __nnx__/__cunnx__ : experimental neural network modules and criterions : `SpatialReSampling`, `SoftMaxTree`, etc. ;
 * __rnn__ : recurrent neural network library. Implements RNN, LSTM, BRNN, BLSTM, and RAM ;
 * __moses__ : utility-belt library for functional programming in Lua, mostly for tables ;
 * __threads__/__parallel__ : libraries for multi-threading or multi-processing ;
 * __async__ : asynchronous library for Lua, inspired by Node.js ;

---

# Tensors

Tensors are the main class of objects used in Torch 7 :

 * An N-dimensional array that views an underlying `Storage` (a contiguous 1D-array);
 * Different Tensors can share the same `Storage`;
 * Different types : `FloatTensor`, `DoubleTensor`, `IntTensor`, `CudaTensor`, and so on ;
 * Implements most Basic Linear Algebra Sub-routines (BLAS) : 
   * `torch.addmm` : matrix-matrix multiplication ;
   * `torch.addmv` : matrix-vector multiplication ;
   * `torch.addr` :  outer-product between vectors ;
   * etc.  
 * Supports random initialization, indexing, transposition, sub-tensor extractions, and more ;
 * Most operations for Float/Double are also implemented for Cuda Tensors (via cutorch) ;

---

## Tensors - Initialization

A `3x2` Tensor  :

```lua
th> a = torch.FloatTensor(3,2)
-- initialized with garbage content (whatever was already there)
th> a 
 8.6342e+19  4.5694e-41  8.6342e+19
 4.5694e-41  0.0000e+00  0.0000e+00
[torch.FloatTensor of size 2x3]
```

Fill with ones :

```lua
th> a:fill(1)
 1  1  1
 1  1  1
[torch.FloatTensor of size 2x3]
```

Random uniform initialization :

```lua
th> a:uniform(0,1) -- random uniform between 0 and 1
 0.6323  0.9232  0.2930
 0.8412  0.5131  0.9101
[torch.FloatTensor of size 2x3]
```

---

## Tensors - Transpose

Let's create a new Tensor `b`, the transpose of dimensions `1` and `2` of Tensor `a` :

```lua
th> b = a:transpose(1,2)
```

Tensor `a` and `b` share the same underlying storage (look at the `1.0000` in both):

```lua
th> b[{1,2}] = 1
th> b
 0.6323  1.0000
 0.9232  0.5131
 0.2930  0.9101
[torch.FloatTensor of size 3x2]

th> a
 0.6323  0.9232  0.2930
 1.0000  0.5131  0.9101
[torch.FloatTensor of size 2x3]
```

---

## Tensors - Storage

This is what the storage looks like :

```lua
th> a:storage()

 0.6323
 0.9232
 0.2930
 1.0000
 0.5131
 0.9101
[torch.FloatStorage of size 6]
```

Yet `a` and `b` have different strides :

```lua
th> unpack(a:stride():totable())
3  1

th> unpack(b:stride():totable())
1  3
```

---

## Tensor - Contiguous

Are `a` and `b` contiguous?

```lua
th> a:isContiguous(), b:isContiguous()
true false
```

Tensor `b` isn't. This means that the elements in the last dimension (the row) aren't contiguous in memory :

```lua
th> b[{1,1}], b[{1,2}], b[{2,1}]
0.63226145505905	1	0.92315602302551	

th> b:storage()
 0.6323 -- 1,1
 0.9232 -- 2,1
 0.2930
 1.0000 -- 1,2
 0.5131
 0.9101
[torch.FloatStorage of size 6]
```

---

## Tensors - Clone/Copy

We can make it contiguous by cloning it :

```lua
th> c = b:clone()
th> c:isContiguous()
true
```

or by copying it :

```lua
th> d = b.new()
th> d:resize(b:size())
th> d:copy(b)
th> d:isContiguous()
true
```

Note : `clone()` allocates memory, while `copy()` doesn't. However, `resize()` sometimes does. 
Above it does because `b.new()` intializes an empty Tensor. 

---

## Tensors - Resize

Calling `resize()` again doesn't allocate new memory (it already has the right size) :

```lua
th> d:resize(b:size())
th> d:storage():size()
6
```

An neither would resizing it to a smaller size :

```lua
th> d:resize(3)
th> d:storage():size()
6
```

But resizing to a greater size will allocate new memory :

```lua
th> e = torch.FloatTensor(d):resize(3,3)
th> e:storage():size() == d:storage():size()
false
```
Tensors `d` and `e` have different storages after the resize.

---

## Tensors - BLAS

.center[![mmm](https://raw.githubusercontent.com/nicholas-leonard/slides/master/matrixmul.png)]

Tensors are all about basic linear algebra. 
Let's multiply an `input` and a `weight` matrix into an `output` matrix :

```lua
batchSize, inputSize, outputSize = 4, 2, 3
input = torch.FloatTensor(batchSize, inputSize):uniform(0,1)
weight = torch.FloatTensor(outputSize, inputSize):unfirom(0,1)
output = torch.FloatTensor()
-- matrix matrix multiply :
output:addmm(0, self.output, 1, input, weight:t())
```

`output` will be automatically resized to `batchSize x outputSize`.
This is a common operation used by the popular `nn.Linear` module.

---

## Tensors - CUDA

Let's what the difference is for doing the previous matrix-matrix multiply using CUDA :

```lua
require 'cutorch'
input = torch.CudaTensor(batchSize, inputSize):uniform(0,1)
weight = torch.CudaTensor(outputSize, inputSize):unfirom(0,1)
output = torch.CudaTensor()
-- matrix matrix multiply :
output:addmm(0, self.output, 1, input, weight:t())
```

So basically, no difference except for use of `torch.CudaTensor`.
However, it will be much faster on GPU than CPU for larger Tensors.
This is especially true for high-end cards like NVIDIA Titan X and such.

---

# Neural Network library

The __nn__ package : 

 * implements feed-forward neural networks ;
 * neural networks form a computational flow-graph of transformations (forward)  ;
 * backpropagation is gradient descent using the chain rule (backward);
 
Two abstract classes :

 * `Module` :  differentiable transformations of input to output ;
 * `Criterion` : cost function to minimize. Outputs a scalar loss;
 
Let's use it to build a simple logistic regressor...

---
 
## Logistic Regression - Module

A binary logisitic regressor `Module` with 2 input units and 1 output.

```lua
require 'nn'
module = nn.Sequential()
module:add(nn.Linear(2, 1))
module:add(nn.Sigmoid())
```

The above implements :

.center[![](https://raw.githubusercontent.com/nicholas-leonard/slides/master/logreg2.png)]

where the sigmoid (logistic function) is defined as :

.center[![](https://raw.githubusercontent.com/nicholas-leonard/slides/master/sigmoid2.png)]

---

## Logistic Regression - Criterion and Data

A binary cross-entropy `Criterion` (which expects 0 or 1 valued targets) :

```lua
criterion = nn.BCECriterion()
```

The BCE loss is defined as :

.center[![](https://raw.githubusercontent.com/nicholas-leonard/slides/master/bce2.png)]

Some random dummy dataset with 10 samples:

```lua
inputs = torch.Tensor(10,2)
targets = torch.Tensor(10):random(0,1)
```

---

## Logistic Regression - Training

Function for one epoch of stochastic gradient descent (SGD)

```lua
require 'dpnn'
function trainEpoch(module, criterion, inputs, targets)
   for i=1,inputs:size(1) do
      local idx = math.random(1,inputs:size(1))
      local input, target = inputs[idx], targets:narrow(1,idx,1)
      -- forward
      local output = module:forward(input)
      local loss = criterion:forward(output, target)
      -- backward
      local gradOutput = criterion:backward(output, target)
      local gradInput = module:backward(input, gradOutput)
      -- update
      module:updateGradParameters(0.9) -- momentum (dpnn)
      module:updateParameters(0.1) -- learning rate
   end
end
```

Do 100 epochs to train the module :

```lua
for i=1,100 do
   trainEpoch(module, criterion, inputs, targets)
end
```

---

# Deep Learning

What is deep learning?
 
 * **collection of techniques** to improve the optimization and **generalization** of neural networks :
  * rectified linear units ;
  * dropout ;
  * batch normalization ;
  * weight decay regularization ;
  * momentum learning ;
 * **stacking layers** of transformations to create successively more abstract **levels of representations** ;
  * depth over breadth ;
  * deep multi-layer perceptrons ;
 * **shared parameters** : 
  * convolutional neural networks ;
  * recurrent neural networks ;
 * **technological improvements** :
  * massively parallel processing : GPUs, CUDA ;
  * fast libraries : torch, cudnn, cuda-convnet, theano ;

---

background-image: url(https://raw.githubusercontent.com/nicholas-leonard/slides/master/we-need-to-go-deeper.jpg)

## Deep Learning - Summary 

---

## Deep Learning - MNIST dataset

.center[![](https://raw.githubusercontent.com/nicholas-leonard/slides/master/mnist.png)]

__dp__ makes it easy to obtain the MNIST dataset :

```lua
require 'dp'
ds = dp.Mnist()
```

Training and validation set inputs and targets :

```lua
trainInputs = ds:get('train', 'inputs', 'bchw')
trainTargets = ds:get('train', 'targets', 'b')
validInputs = ds:get('valid', 'inputs', 'bchw')
validTargets = ds:get('valid', 'targets', 'b')
```

*bchw* specifies axis order : `batch x color x height x width`

---

# Multi-Layer Perceptron 

.center[![](https://raw.githubusercontent.com/nicholas-leonard/slides/master/mlp.png)]

An MLP is a stack of parameterized non-linear layers :

 * each layer is an affine transform (`Linear`) followed by a transfer function (`Tanh`, `ReLU`, `SoftMax`) ;
 * parameters (`weight`, `bias`) are found in the `Linear` module ;
 * transfer functions help to model complex relationships between input and output (non-linear);
 * parameters are varied to fit the data ;

---

## Multi-Layer Perceptron - Module and Criterion

An MLP with 2 layers of hidden units :

```lua
module = nn.Sequential()
module:add(nn.Collapse(3))
module:add(nn.Linear(1*28*28, 200))
module:add(nn.Tanh())
module:add(nn.Linear(200, 200))
module:add(nn.Tanh()) 
module:add(nn.Linear(200, 10))
module:add(nn.LogSoftMax()) -- for classification problems
```

Negative Log-Likelihood (NLL) Criterion :

```lua
criterion = nn.ClassNLLCriterion()
```

---

## Multi-Layer Perceptron - Cross-validation

A function to evaluate performance on the validation set :

```lua
require 'optim'
cm = optim.ConfusionMatrix(10)
function classEval(module, inputs, targets)
   cm:reset()
   for idx=1,inputs:size(1) do
      local input, target = inputs[idx], targets:narrow(1,idx,1)
      local output = module:forward(input)
      cm:add(input, target)
   end
   cm:updateValids()
   print(cm)
   return cm.totalValids
end
```

---

## Multi-Layer Perceptron - Early-Stopping

Early-stopping on the validation set :

```lua
bestAccuracy, bestEpoch = 0, 0
wait = 0
for epoch=1,300 do
   trainEpoch(module, criterion, trainInputs, trainTargets)
   local validAccuracy = classEval(module, validInputs, validTargets)
   if validAccuracy > bestAccuracy then
      bestAccuracy, bestEpoch = validAccuracy, epoch
      torch.save("/path/to/saved/model.t7", module)
      print(string.format("New maxima : %f @ %f", bestAccuracy, bestEpoch))
      wait = 0
   else
      wait = wait + 1
      if wait > 30 then break end
   end
end
```

Early-stops when no new maxima has been found for 30 consecutive epochs.

---

# Convolutional Neural Network

.center[![](https://raw.githubusercontent.com/nicholas-leonard/slides/master/convnet.png)]

Convolutional neural networks are often stacks of 3 layers :

 1. convolution : convolve a kernel over the image along height and width axes ; 
 2. sub-sampling : usually max-pooling, reduces the size (height x width) of feature maps ;
 3. transfer function : a non-linearity like `Tanh` or `ReLU` ;
 

---

## Convolutional Neural Network - Convolution

.center[![](https://raw.githubusercontent.com/nicholas-leonard/slides/master/convolution2.gif)]

Convolution modules typically have the following arguments :

 * `padSize` : how much zero-padding to add around the input image ;
 * `inputSize` : number of input channels (e.g. 3 for RGB image) ;
 * `outputSize` : number of feature maps in kernel ; 
 * `kernelSize` : height and width of the kernel ;
 * `kernelStride` : step-size of the kernel (typically 1) ;

Parameters of the convolution (i.e. the kernel) :
 
 * `weight` : 4D Tensor of size  `outputSize x inputSize x kernelSize x kernelSize` ;
 * `bias` : 1D Tensor of size `outputSize` ;

---

## Convolutional Neural Network - Sub-sampling


---


# Recurrent Neural Network

---


