require 'nn'
require 'dpnn' 
local dl = require 'dataload' -- provides MNIST
require 'optim'

-- options : hyper-parameters and such

local opt = {
   hiddensize = 200,
   batchsize = 32,
   maxepoch = 100,
   learningrate = 0.1,
   momentum = 0.9
}

-- data : MNIST
local trainset, validset, testset = dl.loadMNIST()

-- model : multi-layer perceptron with 2 hidden layers
local mlp = nn.Sequential()
   :add(nn.Convert()) -- casts input to model type (float -> double)
   :add(nn.Collapse(3)) -- collapse 3D to 1D
   :add(nn.Linear(1*28*28, opt.hiddensize))
   :add(nn.Tanh())
   :add(nn.Linear(opt.hiddensize, opt.hiddensize))
   :add(nn.Tanh()) 
   :add(nn.Linear(opt.hiddensize, 10))
   :add(nn.LogSoftMax()) -- for classification problems

-- criterion : binary cross entropy
local nll = nn.ClassNLLCriterion()

-- training : stochastic gradient descent (SGD) with momentum learning
local function trainEpoch(module, criterion, trainset)
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

--[[
Exercise 2 (10 min) : 
  1. take options from the cmd-line (hint : torch.CmdLine())
  2. option to use Dropout in hidden layers (hint : consult nn doc)
  3. Bonus : write script to evaluate saved model on test set (hint : use classEval)
  4. Bonus : make the number of hidden layers a hyper-parameter (hint : for loop)
--]]
