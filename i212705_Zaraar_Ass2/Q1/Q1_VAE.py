import Custom_Dataset
import helper_functions
import VAE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import Transformations as Transformations
import torch
'''
This is a docstring for Code Documentation.
First Directory contains manually annotated signatures from the Previous Assignment.
The Upscaled_and_Renamed_Images will have Images upscaled to the size (224,224)
The Directory will be in the Following Format : 
    Renamed_Signatures ----->     
                            1 (Basically means Signatures of Person Represented by Index 1)
                            ---->1_Image_1.png
                            ---->1_Image_2.png
                            ---->1_Image_3.png
                            ---->1_Image_4.png

                            2 (Basically means Signatures of Person Represented by Index 2)
                            ---->2_Image_1.png
                            ---->2_Image_2.png
                            ---->2_Image_3.png
                            ---->2_Image_4.png
                            .
                            .
                            .
                            and so on
'''
main_directory = 'Signatures'
new_directory = 'Upscaled_and_Renamed_Images'
# Function to upscale the image to 224x224
# helper_functions.upscale_images(main_directory,new_directory)

# DataFrame 
Train_Frame,Test_Frame=helper_functions.train_test_frames(new_directory)

# Train and Test Transformation
train_transform=Transformations.train_transform
test_transform=Transformations.test_transform

# Training and Testing Datasets
Train_Dataset=Custom_Dataset.CustomDataset(Train_Frame,transform=train_transform)
Test_Dataset=Custom_Dataset.CustomDataset(Test_Frame,transform=test_transform)

# Training and Testing Loader
Train_Loader=DataLoader(Train_Dataset, batch_size=16, shuffle=True)
Test_Loader=DataLoader(Test_Dataset, batch_size=16, shuffle=False)

VAE.Train_VAE(Train_Loader)


