class: center, middle

# Torch7: Applied Deep Learning for Vision and Natural Language

Nicholas Leonard

Element Inc.

October 8, 2015

---

# Agenda

1. Introduction - 5 min
2. Packages - 5 min
2. Tensors – 10 min
3. Modules and Criterions – 10 min
4. Training and Evaluation – 10 min
5. Convolutional Neural Networks – 10 min
6. Recurrent Neural Networks – 10 min
7. Hyper-optimization – 5 min

---

# Introduction

My background:
 * was an Army Communications and Electronics Officer for 9 years ;
 * studied in Yoshua Bengio's lab : Python, Theano and Pylearn2 ;
 * switched to Torch7 in 2013 : needed simple modular fast deep learning framework ;
 * employed at a deep learning and biometrics startup called Element Inc ;
 
---

# Introduction - Lua

Why you should take the time to learn Lua?
 * easy interface between low-level C/CUDA/C++ and high-level Lua ;
 * light-weight and extremely powerful ;
 * Tables can be used as lists, dictionaries, classes and objects ;
 * Tables make it easy to extend existing classes (at any level) ;
 * If you like Python, you will likely like Lua ;
 
---

# Introduction - Torch 7

What's up with Torch 7?
 * a scientific computing *distribution* with an emphasis on deep learning ;
 * written in Lua, it has a simple interface to low-level C/CUDA ;
 * neural network modules make it easy to assemble and train MLPs, CNNs, RNNs, etc. ;
 * under development since October 2002 ;
 * used by Facebook, Google [DeepMind], Twitter, NYU, Purdue University, etc.

---

# Packages

The Torch 7 distribution is made up of different packages, each its own github repository :
 * torch7/cutorch : tensors, BLAS, file I/O (serialization), utilities for unit testing and cmd-line argument parsing ;
 * nn/cunn : easy and modular way to build and train simple or complex neural networks using `modules` and `criterions` ;
 * optim : optimization package for nn. Provides training algorithms like SGD, LBFGS, etc. Uses closures ;
 * trepl : torch read–eval–print loop, Lua interpreter, `th>` ;
 * paths : file system manipulation package ;
 * images : for saving, loading, constructing, transforming and displaying images ;

See the torch.ch website for a more complete list of official packages.
 
---

# Packages

Many more non-official packages out there :
 * dp : deep learning library for cross-validation (early-stopping). An alternative to optim inspired by Pylearn2. Lots of documentation and examples ;
 * dpnn : extensions to the nn library. REINFORCE algorithm ;
 * nnx/cunnx : experimental neural network modules and criterions : `SpatialReSampling`, `SoftMaxTree`, etc. ;
 * rnn : recurrent neural network library. Implements RNN, LSTM, BRNN, BLSTM, and RAM ;
 * moses : utility-belt library for functional programming in Lua, mostly for tables ;
 * threads/parallel : libraries for multi-threading or multi-processing ;
 * async : asynchronous library for Lua, inspired by Node.js ;

---

# Tensors

Tensors are the main class of objects used in Torch 7 :
 * An N-dimensional array that views an underlying `Storage` (a contiguous 1D-array);
 * Different Tensors can share the same `Storage`;
 * Different types : `FloatTensor`, `DoubleTensor`, `IntTensor`, `CudaTensor` ;
 * Implements most Basic Linear Algebra Sub-routines (BLAS) : 
   * `torch.addmm` : matrix-matrix multiplication ;
   * `torch.addmv` : matrix-vector multiplication ;
   * `torch.addr` :  outer-product between vectors ;
   * etc.  
 * More goodies : random initialization, indexing, transposition, sub-tensor extractions ;
 * Most operations for Float/Double are also implemented for Cuda Tensors (via cutorch) ;

---

# Tensors - Initialization

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

# Tensors - Transpose

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

# Tensors - Storage

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

# Tensor - Contiguous

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

# Tensors - Clone/Copy

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

# Tensors - Resize

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

# Tensors - BLAS

![mmm](https://raw.githubusercontent.com/nicholas-leonard/slides/master/matrixmul.png)

Tensors are all about basic linear algebra. 
Let's multiply an `input` and a `weight` matrix into an `output` matrix :

```lua
th> batchSize, inputSize, outputSize = 4, 2, 3
th> input = torch.FloatTensor(batchSize, inputSize):uniform(0,1)
th> weight = torch.FloatTensor(outputSize, inputSize):unfirom(0,1)
th> output = torch.FloatTensor()
-- matrix matrix multiply :
th> output:addmm(0, self.output, 1, input, weight:t())
```

`output` will be automatically resized to `batchSize x outputSize`.
This is a common operation used by the popular `nn.Linear` module.

---

# Tensors - CUDA

Let's what the difference is for doing the previous matrix-matrix multiply using CUDA :

```lua
require 'cutorch'
th> input = torch.CudaTensor(batchSize, inputSize):uniform(0,1)
th> weight = torch.CudaTensor(outputSize, inputSize):unfirom(0,1)
th> output = torch.CudaTensor()
-- matrix matrix multiply :
th> output:addmm(0, self.output, 1, input, weight:t())
```

So basically, no difference except for use of `torch.CudaTensor`.
However, it will be much faster on GPU than CPU for larger Tensors.
This is especially true for high-end cards like NVIDIA Titan X and such.

---

# Modules and Criterions

The _nn_ package uses `nn.Modules` and `nn.Criterions` for implementing modular backpropagation. 
Backpropagation is gradient descent using the chain rule.
 
Modules are links in the gradient descent chain. 
Essentially, backpropagation (learning) can be implemented with 3 methods :
 
 * `[output] module:forward(input)` : calls `updateOutput`. Transforms `input` into `output`.
 * `[gradInput] module:backward(input, gradOutput [, scale])` : calls `updateGradInput` followed by `accGradParameters`.  Transforms `gradOutputs` into `gradInputs` and accumulates gradients w.r.t. parameters.
 * `module:updateParametes(lr)` : updates the parameters using the gradients w.r.t. parameters: `param = param - lr*gradParam`. 
 
Criterions measure the `loss` of a neural network (a graph of modules).
They also have `forward`/`backward` methods :

 * `[loss] criterion:forward(input, target)` : measures `loss` (a scalar) given `input` (output of the neural network) and `target` (supervised learning) ;
 * `[gradInput] criterion:backward(input, target)` : generates `gradInput` : gradient of `loss` w.r.t. to `input` ;

---

# Modules and Criterions - SGD

Modules and Criterions implement a differentiable equation that results in a scalar `loss`.
For Stochastic Gradient Descent (SGD), the loss is minimized by obtaining
the gradients w.r.t. the learnable parameters of the modules, 
and moving the parameters in the opposite direction of these gradients. 
These parameter updates are scaled by a small learning rate `lr`, usually between `0.1 and 0.0001`. 

Stochastic gradient descent is the most common training algorithm. 
It can be augmented with momentum, which involves 



