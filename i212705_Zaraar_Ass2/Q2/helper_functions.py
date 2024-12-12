import os
import torchvision
import torchvision.transforms as transforms

def create_data(cat_dir,dog_dir):
    # Create directories if they don't exist
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # CIFAR-10 class indices
    # 3 - Cat, 5 - Dog
    classes = trainset.classes
    cat_label = classes.index('cat')
    dog_label = classes.index('dog')

    # Extract and save cat and dog images
    for i, (image, label) in enumerate(trainset):
        # Convert tensor to a PIL image
        image = transforms.ToPILImage()(image)
        
        # Save the image in the respective directory based on the label
        if label == cat_label:
            image.save(os.path.join(cat_dir, f'cat_{i}.png'))
        elif label == dog_label:
            image.save(os.path.join(dog_dir, f'dog_{i}.png'))

    print("Images of cats and dogs have been saved in separate directories.")
