require 'nn'
require 'dpnn' 
local dl = require 'dataload' -- provides MNIST
require 'optim'

-- options : hyper-parameters and such
cmd = torch.CmdLine() -- 1 take options from cmd-line
cmd:text()
cmd:text('Image Classification using Multi-Layer Perceptron (training)')
cmd:text('Options:')
cmd:option('-modelpath', paths.concat(dl.SAVE_PATH, "mlp.t7"), 'path to saved model')
local opt = cmd:parse(arg or {})

-- data : MNIST
local trainset, validset, testset = dl.loadMNIST()

-- model
local mlp = torch.load(opt.modelpath)

-- test set evaluation
local cm = optim.ConfusionMatrix(10) 
function classEval(module, validset) -- same as in train-mlp.lua
   module:evaluate()
   cm:zero()
   for i, input, target in validset:subiter(opt.batchsize) do
      local output = module:forward(input)
      cm:batchAdd(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end

local testAccuracy = classEval(mlp, testset)
print(string.format("Test accuracy=%f", testAccuracy))
