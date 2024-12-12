import Transformations
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, random_split
import SN_GAN
cat_dir='cats'
dog_dir='dogs'
# helper_functions.create_data(cat_dir,dog_dir)

# Load dataset (combine cats and dogs)
cat_dataset = datasets.ImageFolder(root='dataset', transform=Transformations.train_transform)
dog_dataset = datasets.ImageFolder(root='dataset', transform=Transformations.train_transform)

# Combine the two datasets into one
dataset = ConcatDataset([cat_dataset, dog_dataset])

# Calculate the sizes for training and testing
total_size = len(dataset)
test_size = int(0.1 * total_size)  # 10% for testing
train_size = total_size - test_size  # Remaining 90% for training

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
Train_Loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
Test_Loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


SN_GAN.Train_SN_GAN(Train_Loader)

