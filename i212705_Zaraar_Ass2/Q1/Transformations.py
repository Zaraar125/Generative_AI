import torchvision.transforms as transforms

# train_transform = transforms.Compose([
#     transforms.RandomRotation(degrees=(-30, 30)),
#     transforms.ColorJitter(brightness=0.5),
#     transforms.RandomResizedCrop(size=(64, 128), scale=(0.8, 1.2)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

train_transform=transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(64, 128), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

GAN_train_transform=transforms.Compose([

    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=15)),
    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=-15)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(64, 128), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    
])
# train_transform=transforms.Compose([transforms.RandomHorizontalFlip(),
#     transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=15)),
#     transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=-15)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.CenterCrop((64,128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    
# ])



test_transform = transforms.Compose([
    transforms.Resize((64, 128)),  # Resize to 224x224 (no crop for testing)
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])