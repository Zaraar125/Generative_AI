import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader,Dataset
from torch.nn.functional import one_hot
z=r'sign\r'
class CustomDataset(Dataset):
    def __init__(self,frame,num_classes):

        self.classes=num_classes
        self.frame=frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        image=self.frame.iloc[idx][1]
        label=self.frame.iloc[idx][0]
        # print(input)

        image=z+image[1:]
        # print(input)
        # print(image)
        input=cv2.imread(image,0)
        # print(input)

        input=torch.tensor(input)
        input=torch.flatten(input)[:4096]
        # print(input.shape)
        label=torch.tensor(int(label))
        label=one_hot(label,self.classes)
        label=torch.tensor(label,dtype=torch.float32)
        
        return input,label