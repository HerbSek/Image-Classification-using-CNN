import torch
from torch import nn

## CNN model in here 
class model_y(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_conv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),  # Pooling layers 

            # Second convolutional layer
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),  # Normalize features in hidden layers 
            nn.ReLU(),    
            nn.MaxPool2d(2,2),  # Pooling layers 

            # Third convolutional layer
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),  # Pooling layers 

            nn.Flatten() # Convert from Conv2d to Linear 
        )

        self.full_fcl = nn.Sequential(
             # Fully connected layers below 
            nn.Linear(64*28*28, 128), # 128 is a hyperparameter used n FCL as a design choice !!!
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,20) # Output layer consists of '20' neurons 
        )

    def forward(self, x):
        y = self.full_conv(x)
        return self.full_fcl(y)

    
    



