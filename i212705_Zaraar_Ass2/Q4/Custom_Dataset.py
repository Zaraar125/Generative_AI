from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class PersonSketchDataset(Dataset):
    def __init__(self, root_person, root_sketch, transform=None):
        
        self.root_person = root_person
        self.root_sketch = root_sketch
        
        self.transform = transform

        self.person_images = os.listdir(root_person)
        self.sketch_images = os.listdir(root_sketch)
        
        self.length_dataset = max(len(self.person_images), len(self.sketch_images))
        
        self.person_images_len = len(self.person_images)
        self.sketch_images_len = len(self.sketch_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        person_img = self.person_images[index % self.person_images_len]
        sketch_img = self.sketch_images[index % self.sketch_images_len]

        person_path = os.path.join(self.root_person, person_img)
        sketch_path = os.path.join(self.root_sketch, sketch_img)

        person_img = np.array(Image.open(person_path).convert("RGB"))
        sketch_img = np.array(Image.open(sketch_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=person_img, image0=sketch_img)
            person_img = augmentations["image"]
            sketch_img = augmentations["image0"]

        return person_img, sketch_img