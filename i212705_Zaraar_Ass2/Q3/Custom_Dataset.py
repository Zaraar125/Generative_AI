import cv2
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):    
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Retrieve the sample at the specified index."""
        # Get the image path from the DataFrame
        face,sketch = self.dataframe.iloc[idx]['Faces'],self.dataframe.iloc[idx]['Sketches']
        face=cv2.imread(face)
        sketch= cv2.imread(sketch)

        face= cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        sketch=cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

        face=cv2.resize(face,(224,224))
        sketch=cv2.resize(sketch,(224,224))

        face=face/255          # Normalization
        sketch=sketch/255

        face = face * 2 - 1
        sketch = sketch * 2 - 1

        face=torch.tensor(face,dtype=torch.float32)
        sketch=torch.tensor(sketch,dtype=torch.float32)

        return face,sketch



