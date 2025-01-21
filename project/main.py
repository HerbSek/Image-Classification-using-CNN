from torch import nn
import torch.optim as optim

# Using sequential containers generally !!!
model = nn.Sequential(
    nn.Linear(input_, hidden_),
    nn.ReLU(),
    nn.Linear(hidden_, output_),
    nn.Sigmoid()
)

loss_funct = nn.MSELoss() # nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.01) # Adam 

# Iteration #

optimizer.zero_grad() # discards gradients after tests have been made from previous iterations and the step !!!

y_pred =  model(x) # x is input data to predict (target matrix)

loss = loss_funct(y_pred, y) # predicted y value with the actual value .
# The loss function calculates how worse the model performs. If the loss function converges then model is learning !!!

loss.backward() # calculates loss function


optimizer.step() # updates parameters for next iteration !!!