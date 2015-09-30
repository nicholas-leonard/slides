class: center, middle

# Torch7: Applied Deep Learning for Vision and Natural Language

Nicholas Leonard

Element Inc.

October 8, 2015

---

# Agenda

1. Tensors – 5 min
2. Modules and Criterions – 10 min
3. Training and Evaluation – 10 min
4. Convolutional Neural Networks – 10 min
5. Recurrent Neural Networks – 10 min
6. Hyper-optimization – 5 min

---

# Tensors

A `3x2` Tensor initialized with random scalars (sometimes NaNs).
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
