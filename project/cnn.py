import torch
from torch import nn


class model_y(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_conv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            # Second convolutional layer
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),    
            nn.MaxPool2d(2,2),  

            # Third convolutional layer
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

    def calc_fcl(self,x):
        return self.full_conv



