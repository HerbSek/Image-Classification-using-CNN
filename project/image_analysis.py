from PIL import Image  
import PIL
from PIL.Image import Image as img
import numpy as np
import os
import torch
from cnn import model_y
# Code to get fcl below 

image_path  = fr"C:\Users\Herbert\OneDrive\Desktop\Corn Disease\data\Blight\Corn_Blight (3).jpg"

def get_tensor(image_path):
    image_path = Image.open(image_path)
    img_size = (224,224)
    resized_img = image_path.resize(img_size)
    numpy_img = np.array(resized_img, dtype = np.float32)
    torch_img = torch.tensor(numpy_img, dtype = torch.float32)
    return torch_img


if __name__ == "__main__":
    output_tensor = get_tensor(image_path)
    output_tensor = output_tensor.unsqueeze(0)
    # print(output_tensor)
    output_tensor = output_tensor.permute([0,3,2,1])
    print(f'Initial shape before CNN : {output_tensor.shape}')
    # print(output_tensor.shape)
    my_model = model_y()
    input_data = my_model.full_conv(output_tensor)
    print(f'Final shape to pass to Fully connected layer : {input_data.shape}')
