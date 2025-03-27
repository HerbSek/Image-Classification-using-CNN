# Single layer neural Network Architecture 
import torch 
from torch import nn,optim
import random

random.seed(42)


class model_x(nn.Module):
    def __init__(self):
        super().__init__() # Initialize nn.Module
        self.model = nn.Sequential(
            nn.Linear(5,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
        
    def train_model(self,x):
        y_pred = self.model(x)
        return y_pred
    

my_model = model_x()
    
x = torch.randn(100,5)   

y = torch.randint(0,2, (100,1), dtype = torch.float32)


optimizer = optim.SGD(my_model.parameters(), lr = 0.01) 
loss_function = nn.MSELoss() 


count = 0
for i in range(100):
    optimizer.zero_grad()
    output = my_model.train_model(x)
    loss = loss_function(output , y ) # y is the ground truth !!!
    loss.backward() # Calculate loss function !!!
    optimizer.step() # Step learning process by an epoch !!!
    count = count+1
    print(f" epoch: {count} , loss function: {loss}")


print("Model done training ") 


