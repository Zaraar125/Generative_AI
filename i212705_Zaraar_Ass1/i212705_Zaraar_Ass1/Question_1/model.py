import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # Assuming input has 1 channel (e.g., grayscale)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling Layer
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu=nn.ReLU()
        # Fully Connected Layer
        self.fc = nn.Linear(32 * 7 * 7, num_classes)  # Assuming input images are 28x28 (e.g., MNIST)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x)))  
        x = x.view(-1, 32 * 7 * 7)  
        x = self.fc(x)  
        return x
    
def train_test(train_loader):
    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 16
    num_classes = 10  # Example for classification (e.g., digits 0-9)

    # Instantiate the model, loss function, and optimizer
    model = CNNModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

     # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            

            outputs = model(inputs)  # Forward pass
            print(outputs.shape)
            print(inputs.shape)
            # break
            loss = criterion(outputs, targets)  # Calculate loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            total_loss += loss.item()  # Accumulate loss

        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
    return model

