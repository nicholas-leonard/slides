require 'nn'
require 'dpnn' -- provides nn.Module:updateGradParameters(momentum)

-- data : dummy data
inputs = torch.Tensor(100,2):uniform(-1,1)
targets = torch.Tensor():gt(inputs:select(2,1) + inputs:select(2,2) , 0) + 1 -- class 1 if sum of inputs <= 0, class 2 otherwise

-- model : logistic regressor
local logreg = nn.Sequential()
logreg:add(nn.Linear(2, 1))
logreg:add(nn.Sigmoid())

-- criterion : binary cross entropy
local bce = nn.BCECriterion()

-- training : stochastic gradient descent (SGD) with momentum learning
local function trainEpoch(module, criterion, inputs, targets)
   for i=1,inputs:size(1) do
      local idx = math.random(1,inputs:size(1))
      local input, target = inputs[idx], targets:narrow(1,idx,1)
      -- forward
      local output = module:forward(input)
      local loss = criterion:forward(output, target)
      -- backward
      local gradOutput = criterion:backward(output, target)
      module:zeroGradParameters()
      local gradInput = module:backward(input, gradOutput)
      -- update
      module:updateGradParameters(0.9) -- momentum (dpnn)
      module:updateParameters(0.1) -- W = W - 0.1*dL/dW
   end
end

trainEpoch(logreg, bce, inputs, targets)

--[[
Exercise 1 (10 min) : 
  1. modify the above to train for 100 epochs (hint : for loop)
  2. each call to trainEpoch prints the mean error of that epoch (hint : print)
  3. bonus : modify the above to use Softmax + ClassNLLCriterion (hint : modify data, model, criterion)
--]]
