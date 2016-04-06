require 'nn'
require 'dpnn' 
local dl = require 'dataload' -- provides MNIST
require 'optim'

-- options : hyper-parameters and such
cmd = torch.CmdLine() 
cmd:text()
cmd:text('Image Classification using Convolutional Neural Network (training)')
cmd:text('Example')
cmd:text("th convolution-neural-network.lua -channelsize '{16,32,32}' -learningrate 0.01")
cmd:text('Options:')
cmd:option('-hiddensize', 200, 'number of hidden units')
cmd:option('-nlayer', 2, 'number of hidden layers')
cmd:option('-batchsize', 32, 'number of examples per batch')
cmd:option('-maxepoch', 200, 'stop experiment after this many epochs')
cmd:option('-learningrate', 0.1, 'SGD learning rate')
cmd:option('-momentum', 0.9, 'SGD momemtum learning factor')
cmd:option('-dropout', false, 'apply dropout on conv and hidden layers. Do not use with batchnorm.')
cmd:option('-batchnorm', false, 'apply batchnormalization on conv and hidden layers. Do not use with dropout.')
cmd:option('-channelsize', '{16,32}', 'Number of output channels for each convolution layer.')
cmd:option('-kernelsize', '{5,5,5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('-kernelstride', '{1,1,1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('-poolsize', '{2,2,2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('-poolstride', '{2,2,2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('-padding', false, 'add math.floor(kernelSize/2) padding to the input of each convolution') 
cmd:option('-cuda', false, 'use CUDA instead of C, i.e. GPU instead of CPU')
local opt = cmd:parse(arg or {})
opt.channelsize = loadstring(" return "..opt.channelsize)()
opt.kernelsize = loadstring(" return "..opt.kernelsize)()
opt.kernelstride = loadstring(" return "..opt.kernelstride)()
opt.poolsize = loadstring(" return "..opt.poolsize)()
opt.poolstride = loadstring(" return "..opt.poolstride)()

-- data : MNIST
local trainset, validset, testset = dl.loadMNIST()

-- model : cnn
local cnn = nn.Sequential()
   :add(nn.Convert())

-- convolutional and pooling layers
local inputsize = 1
for i=1,#opt.channelsize do
   if opt.dropout and i > 1 then
      -- dropout can be useful for regularization
      cnn:add(nn.SpatialDropout(0.5))
   end
   cnn:add(nn.SpatialConvolution(
      inputsize, opt.channelsize[i], 
      opt.kernelsize[i], opt.kernelsize[i], 
      opt.kernelstride[i], opt.kernelstride[i],
      opt.padding and math.floor(opt.kernelsize[i]/2) or 0
   ))
   if opt.batchnorm then
      -- 2 batch normalization can be awesome
      cnn:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
   end
   cnn:add(nn.ReLU())
   if opt.poolsize[i] and opt.poolsize[i] > 0 then
      -- 1 max pooling
      cnn:add(nn.SpatialMaxPooling(
         opt.poolsize[i], opt.poolsize[i], 
         opt.poolstride[i] or opt.poolsize[i], -- stride defaults to size
         opt.poolstride[i] or opt.poolsize[i]
      ))
   end
   inputsize = opt.channelsize[i]
end
-- get output size of convolutional layers
local insize = trainset:isize() -- input size excluding batch dimension : {1,28,28} 
table.insert(insize, 1, 1) -- to batch of one example : {1,1,28,28}
local outsize = cnn:outside(insize) -- equivalent to : cnn:forward(torch.randn(unpack(insize))):size()
local inputsize = outsize[2]*outsize[3]*outsize[4]
print("input to dense layers has: "..inputsize.." units")

-- dense hidden layers
cnn:add(nn.Collapse(3))
for i=1,opt.nlayer do 
   if opt.dropout then
      cnn:add(nn.Dropout(0.5))
   end
   cnn:add(nn.Linear(inputsize, opt.hiddensize))
   if opt.batchnorm then
      -- 2 batch normalization
      cnn:add(nn.BatchNormalization(hiddenSize))
   end
   cnn:add(nn.ReLU())
   inputsize = opt.hiddensize
end

if opt.dropout then
   cnn:add(nn.Dropout(0.5))
end
cnn:add(nn.Linear(inputsize, 10))
cnn:add(nn.LogSoftMax()) -- for classification problems

-- 3 efficient serialization (see dpnn doc)
cnn = nn.Serial(cnn) -- Serial decorates the cnn module
cnn:mediumSerial() -- doesn't serialize gradInput and output module attributes

-- criterion : binary cross entropy
local nll = nn.ClassNLLCriterion()

if opt.cuda then
   require 'cutorch' -- provides torch.CudaTensor and wraps CUBLAS, etc.
   require 'cunn' -- provides the CUDA version for most of the nn modules.
   cnn:cuda() -- cast module to cuda
   nll = nn.ModuleCriterion(nll, nil, nn.Convert()) -- float -> cuda
   nll:cuda() -- cast criterion to cuda
end

-- training : stochastic gradient descent (SGD) with momentum learning
local function trainEpoch(module, criterion, trainset)
   module:training() 
   local sumloss = 0
   for i, input, target in trainset:sampleiter(opt.batchsize) do
      -- forward
      local output = module:forward(input)
      local loss = criterion:forward(output, target)
      sumloss = sumloss + loss*input:size(1)
      -- backward
      local gradOutput = criterion:backward(output, target)
      module:zeroGradParameters()
      local gradInput = module:backward(input, gradOutput)
      -- update
      module:updateGradParameters(0.9) -- momentum (dpnn)
      module:updateParameters(0.1) -- W = W - 0.1*dL/dW
   end
   return sumloss / trainset:size()
end

-- evaluation : cross-validation
local cm = optim.ConfusionMatrix(10)
function classEval(module, validset)
   module:evaluate() 
   cm:zero()
   for i, input, target in validset:subiter(opt.batchsize) do
      local output = module:forward(input)
      cm:batchAdd(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end

-- early-stopping : save the best version of the model on validation set
paths.mkdir(dl.SAVE_PATH)
opt.savepath = paths.concat(dl.SAVE_PATH, "cnn.t7")
opt.bestAccuracy, opt.bestEpoch = 0, 0
local wait = 0
for epoch=1,opt.maxepoch do
   local a = torch.Timer()
   local trainloss = trainEpoch(cnn, nll, trainset)
   print(string.format("Epoch #%d : mean training loss=%f", epoch, trainloss))
   print(string.format("Training speed %f examples/second", trainset:size()/a:time().real))
   
   local validAccuracy = classEval(cnn, validset)
   print(string.format("Validation accuracy=%f", validAccuracy))
   if validAccuracy > opt.bestAccuracy then
      opt.bestAccuracy, opt.bestEpoch = validAccuracy, epoch
      cnn.opt = opt -- save the options with the model
      torch.save(opt.savepath, cnn)
      print(string.format("New maxima saved at %s", opt.savepath))
      wait = 0
   else
      wait = wait + 1
      if wait > 30 then break end
   end
   print""
end
