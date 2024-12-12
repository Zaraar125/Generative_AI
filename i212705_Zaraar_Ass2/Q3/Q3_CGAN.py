import helper_functions
import cv2
import torch
import Custom_Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import c_GAN
# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("almightyj/person-face-sketches")
# print("Path to dataset files:", path)
test_directory= r'Q3\dataset\test'
train_directory = r'Q3\dataset\train'
validation_directory = r'Q3\dataset\val'

# Function to upscale the image to 224x224
# helper_functions.upscale_images(main_directory,new_directory)

# DataFrame 
Train_Frame,Validation_Frame,Test_Frame=helper_functions.create_frames(train_directory,validation_directory,test_directory)

# Dataset
Train_data,Val_data,Test_data=Custom_Dataset.CustomDataset(Train_Frame),Custom_Dataset.CustomDataset(Validation_Frame),Custom_Dataset.CustomDataset(Test_Frame)

# Data loaders
Train_loader,Val_loader,Test_loader=DataLoader(Train_data,shuffle=True,batch_size=8),DataLoader(Val_data,shuffle=True,batch_size=8),DataLoader(Test_data,shuffle=False,batch_size=8)

c_GAN.train_CGAN(Train_loader)