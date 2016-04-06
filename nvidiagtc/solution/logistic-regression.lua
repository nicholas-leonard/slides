require 'nn'
require 'dpnn' -- provides nn.Module:updateGradParameters(momentum)

-- data : dummy data
inputs = torch.Tensor(100,2):uniform(-1,1)
targets = torch.gt(inputs:select(2,1) + inputs:select(2,2) , 0) + 1 -- class 1 if sum of inputs <= 0, class 2 otherwise

-- model : logistic regressor
local logreg = nn.Sequential()
logreg:add(nn.Linear(2, 2)) -- 3 Linear has two outputs for softmax
logreg:add(nn.LogSoftMax()) -- 3 ClassNLLCriterion expects Log of SoftMax

-- criterion : binary cross entropy
local nll = nn.ClassNLLCriterion()

-- training : stochastic gradient descent (SGD) with momentum learning
local function trainEpoch(module, criterion, inputs, targets)
   local sumloss = 0 -- 2
   for i=1,inputs:size(1) do
      local idx = math.random(1,inputs:size(1))
      local input, target = inputs[idx], targets:narrow(1,idx,1)
      -- forward
      local output = module:forward(input)
      local loss = criterion:forward(output, target)
      sumloss = sumloss + loss -- 2
      -- backward
      local gradOutput = criterion:backward(output, target)
      module:zeroGradParameters()
      local gradInput = module:backward(input, gradOutput)
      -- update
      module:updateGradParameters(0.9) -- momentum (dpnn)
      module:updateParameters(0.1) -- W = W - 0.1*dL/dW
   end
   return sumloss / inputs:size(1) -- 2
end

for epoch=1,100 do -- 1
   local meanloss = trainEpoch(logreg, nll, inputs, targets)
   print(string.format("Epoch %d : mean loss = %f", epoch, meanloss))
end

