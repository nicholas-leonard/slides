class: center, middle

# Torch7: Applied Deep Learning for Vision and Natural Language

Nicholas Leonard

Element Inc.

October 8, 2015

---

# Agenda

1. Introduction - 5 min
2. Tensors – 5 min
3. Modules and Criterions – 10 min
4. Training and Evaluation – 10 min
5. Convolutional Neural Networks – 10 min
6. Recurrent Neural Networks – 10 min
7. Hyper-optimization – 5 min

---

# Introduction 

My background:
 * was a Army Communications and Electronics Officer in the Canadian Forces for 9 years;
 * did my masters in Yoshua Bengio's lab working with Python, Theano and Pylearn2;
 * switched to Torch7 in 2013 to work on distributed conditional computation;
 * currently work in NYC at a deep learning and biometrics startup called Element Inc;
 * passion for elegant, modular, simple and fast deep learning code : Torch7/CUDA;

Why should I take the time to learn Lua :
 * easy interface between low-level C/CUDA/C++ and high-level Lua ;
 * Lua is light-weight and extremely powerful ;
 * Tables can be used as lists, dictionaries, classes and objects ;
 * Tables make it easy to extend existing classes (at any level) ;
 * If you like Python, you will like Lua

What's up with Torch7?
 * a scientific computing distribution with an emphasis on deep learning;

---

# Tensors

A `3x2` Tensor initialized with random scalars (sometimes NaNs) :

```lua
th> a = torch.FloatTensor(3,2)
th> a -- initialized with garbage content
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
th> a:uniform(0,1) -- random 
 0.6323  0.9232  0.2930
 0.8412  0.5131  0.9101
[torch.FloatTensor of size 2x3]
```

---

# Tensors

---
