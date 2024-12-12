from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Custom dataset class for loading images for VAE training/testing.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing image paths.
        transform (callable, optional): Optional transform to be applied on an image.
    """
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Retrieve the sample at the specified index."""
        # Get the image path from the DataFrame
        img_path = self.dataframe.iloc[idx]['image_path']
        
        # # Open the image
        # image = cv2.imread(img_path)  # Convert to RGB if necessary
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.open(img_path).convert('RGB')
        # image = image.convert('L')
        # Apply the transformation if provided
        if self.transform:
            image = self.transform(image)

        return image



