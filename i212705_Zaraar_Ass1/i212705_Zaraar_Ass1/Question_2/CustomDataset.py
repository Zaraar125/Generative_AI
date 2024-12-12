import torch
import helper_functions
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

class CustomDataset(Dataset):
    def __init__(self, helper_dict,x,y):

        self.x=x
        self.y=y
        self.helper_dict=helper_dict

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        INPUT = self.x[idx]
        OUTPUT = self.y[idx]

        y_one_hot=torch.tensor(OUTPUT)
        # print(y_one_hot)
        y_one_hot = one_hot(y_one_hot, num_classes=len(self.helper_dict.keys()))

        pad_x=torch.tensor(INPUT,dtype=torch.long)
        y_one_hot=torch.tensor(y_one_hot,dtype=torch.float32)
        
        return pad_x, y_one_hot