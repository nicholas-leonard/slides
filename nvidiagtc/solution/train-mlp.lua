require 'nn'
require 'dpnn' 
local dl = require 'dataload' -- provides MNIST
require 'optim'

-- options : hyper-parameters and such
cmd = torch.CmdLine() -- 1 take options from cmd-line
cmd:text()
cmd:text('Image Classification using Multi-Layer Perceptron (training)')
cmd:text('Options:')
cmd:option('-hiddensize', 200, 'number of hidden units')
cmd:option('-nlayer', 2, 'number of hidden layers')
cmd:option('-batchsize', 32, 'number of examples per batch')
cmd:option('-maxepoch', 100, 'stop experiment after this many epochs')
cmd:option('-learningrate', 0.1, 'SGD learning rate')
cmd:option('-momentum', 0.9, 'SGD momemtum learning factor')
cmd:option('-dropout', false, 'apply dropout on hidden neurons')
local opt = cmd:parse(arg or {})

-- data : MNIST
local trainset, validset, testset = dl.loadMNIST()

-- model : multi-layer perceptron with 2 hidden layers
local mlp = nn.Sequential()
   :add(nn.Convert()) -- casts input to model type (float -> double)
   :add(nn.Collapse(3)) -- collapse 3D to 1D

local inputsize = 1*28*28
for i=1,opt.nlayer do -- 4 number of hidden layers is a hyper-param
   mlp:add(nn.Linear(inputsize, opt.hiddensize))
   mlp:add(nn.Tanh())
   if opt.dropout then -- 2 dropout goes after transfer function
      mlp:add(nn.Dropout(0.5))
   end
   inputsize = opt.hiddensize
end

mlp:add(nn.Linear(inputsize, 10))
mlp:add(nn.LogSoftMax()) -- for classification problems

-- criterion : binary cross entropy
local nll = nn.ClassNLLCriterion()

-- training : stochastic gradient descent (SGD) with momentum learning
local function trainEpoch(module, criterion, trainset)
   module:training() -- 2 training mode (for dropout and batch norm)
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
   module:evaluate() -- 2 evaluate mode (for dropout and batch norm)
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
opt.savepath = paths.concat(dl.SAVE_PATH, "mlp.t7")
opt.bestAccuracy, opt.bestEpoch = 0, 0
local wait = 0
for epoch=1,opt.maxepoch do
   local trainloss = trainEpoch(mlp, nll, trainset)
   print(string.format("Epoch #%d : mean training loss=%f", epoch, trainloss))
   
   local validAccuracy = classEval(mlp, validset)
   print(string.format("Validation accuracy=%f", validAccuracy))
   if validAccuracy > opt.bestAccuracy then
      opt.bestAccuracy, opt.bestEpoch = validAccuracy, epoch
      mlp.opt = opt -- save the options with the model
      torch.save(opt.savepath, mlp)
      print(string.format("New maxima saved at %s", opt.savepath))
      wait = 0
   else
      wait = wait + 1
      if wait > 30 then break end
   end
   print""
end
