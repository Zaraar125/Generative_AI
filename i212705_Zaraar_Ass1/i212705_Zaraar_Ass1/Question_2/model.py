import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)  
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        # Dropout layer: Applied to the output of the LSTM
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):

        x = self.embedding(x)        
        output, (h1, c1) = self.lstm(x)
        output=output[:, -1, :]
        output=self.batch_norm(output)
        output=self.dropout(output)
        out = self.fc(output)
        return out

def train_test(helper_dict,train_loader,val_loader):
    vocab_size = len(helper_dict)  
    embed_size = 32
    hidden_size = 64
    num_layers = 2
    num_epochs = 10
    device='cuda'
    
    # Initialize the model, loss function, and optimizer
    model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    training_epoch_loss = []
    validation_epoch_loss = []
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0   
        model.train()  # Set the model to training mode
        
        # Initialize progress bar for the current epoch
        with tqdm(total=len(train_loader), desc=f'Training Epoch {epoch + 1}/{num_epochs}',ncols=100) as pbar:
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)  
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()  
                pbar.update(1)  # Increment the progress bar

        # Calculate and display the average training loss for the epoch
        average_train_loss = total_train_loss / len(train_loader)  
        training_epoch_loss.append(average_train_loss)

         # Validation phase with tqdm
        model.eval()  # Set the model to evaluation mode
        validation_epoch_loss.append(0)  # Initialize for validation
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'Validation Epoch {epoch + 1}/{num_epochs}',ncols=100) as pbar:
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    total_val_loss += val_loss.item()
                    pbar.update(1)  # Increment the progress bar

            average_val_loss = total_val_loss / len(val_loader)
            validation_epoch_loss.append(average_val_loss) # Update the last entry with average validation loss
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]    Average Training Loss: {average_train_loss:.4f}    Validation Loss: {average_val_loss:.4f}','\n')

    return model,average_train_loss,average_val_loss
    
