import os
import cv2
import torch 
import numpy as np
import pandas as pds
import helper_functions
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader,Dataset
import model

import random
import pandas as pd
path=r'i212705_Zaraar_Ass1\Question_1\Data'
saving_directory='sign'
z=r'\r'

temp=os.listdir(saving_directory)
num_classes=len(temp)
# training and testing data
testing_data=random.sample(temp,10)
testing_labels=[i.split('_')[1].split('.')[0] for i in testing_data]
training_data=[i for i in temp if i not in testing_data]
training_labels=[i.split('_')[1].split('.')[0] for i in training_data]

train=pd.DataFrame([training_labels,training_data]).T
test=pd.DataFrame([testing_labels,testing_data]).T

TRAIN=CustomDataset(train,num_classes)
TEST=CustomDataset(test,num_classes)

train_loader=DataLoader(TRAIN, batch_size=32, shuffle=True)  
test_loader=DataLoader(TEST, batch_size=16, shuffle=False)


Model=model.train_test(train_loader)




