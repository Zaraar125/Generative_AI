import torch
import model
import spacy
import numpy as np
import json
import pandas as pd
import pickle
from CustomDataset import CustomDataset
import helper_functions
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# Dataset File Path
dataset_path=r'i212705_Zaraar_Ass1\Question_2\archive\Shakespeare_data.csv'


# # read the Shakespear Dataset 
a=helper_functions.read_data(dataset_path).to_list()

# Preprocess the data and only keep "PlayerLine" for Training and Testing
s=helper_functions.preprocess(a)               
DICT,new_s=helper_functions.vector_dict(s)
inverse_dict=helper_functions.inverse_dict(DICT)
json.dump( DICT, open( "vector_dict.json", 'w' ) )
json.dump(inverse_dict,open('inverse_dict.json','w'))

# Read data from file:

# print(new_s[0])
# # Generate Ngram Sequence for Training and Testing the Model 
# # # returns a list of tokenized strings
x,y=helper_functions.generate_ngram_sequence(new_s)     
# print(x[:10])

# Generate Unique Word Dictionary for Word-Vectors
# DICT=helper_functions.vector_dict(x,y)

num_x=helper_functions.convert_text_to_vectors(DICT,x)
num_y=helper_functions.convert_TOV_y(DICT,y)
pad_x=helper_functions.pad(num_x,DICT)

x_train, x_test, y_train, y_test = train_test_split(pad_x, num_y, test_size=0.2, shuffle=True, random_state=34)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=34)

train_dataset = CustomDataset(DICT,x_train,y_train)
val_dataset = CustomDataset(DICT,x_val,y_val)
# test_dataset = CustomDataset(DICT,x_test,y_test)

# # # # # Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

# # print('here')
# # model=LSTMModel(DICT,train_loader,test_loader)
MODEL,t_loss,v_loss=model.train_test(DICT,train_loader,val_loader)

# model.train_test(DICT,train_loader,val_loader)

with open('lstm_model.pkl', 'wb') as f:
    pickle.dump(MODEL, f)



