import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.utils as vutils
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) Class.
    """
    
    def __init__(self, input_dim=3, hidden_dim=32, latent_dim=64):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (32, 64, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),  # (16, 32, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),  # (8, 16, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),  # (4, 8, 128)
            nn.ReLU(),
        )

        # Calculate the flattened size after the encoder layers
        self.flattened_size = hidden_dim * 4 * 4 * 8  # (4 * 8 = 32, hidden_dim * 4 = 128)

        # Linear layers for latent variables
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),  # (8, 16, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),  # (16, 32, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (32, 64, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=4, stride=2, padding=1),  # (64, 128, 3)
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )

    def encode(self, x):
        """Encode input into latent space."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode latent vector back to image space."""
        z = self.decoder_input(z)
        z = z.view(z.size(0), -1, 4, 8)  # Reshape for decoder
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Loss function combining reconstruction loss and KL divergence."""
        BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 3), x.view(-1, 3), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def show_reconstructed_image(self, original, reconstructed):
        """Show original and reconstructed images."""
        original = original.cpu().detach().numpy().transpose(1, 2, 0)
        reconstructed = reconstructed.cpu().detach().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed)
        plt.title("Reconstructed Image")
        plt.axis('off')

        plt.show()
            
# Example usage of VAE
def Train_VAE(Train_Loader,save_dir='VAE_Training_Outputs'):
    device='cuda'
    # Instantiate the VAE and move it to the GPU
    vae = VAE(input_dim=3, hidden_dim=32, latent_dim=64).to('cuda')

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    epochs=600
    avg_loss=[]
    for epoch in range(epochs):  # Training for 10 epochs
        running_loss=0
        for batch_idx, data in enumerate(Train_Loader):
            data = data.to('cuda')  # Move data to GPU

            optimizer.zero_grad()
            
            recon_batch, mu, logvar = vae(data)
            loss = vae.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            running_loss=running_loss+loss.item()
        # Display one reconstructed image after each epoch
        print('Epoch : ',epoch,'  Avg Loss : ',running_loss/len(Train_Loader))
        avg_loss.append(running_loss/len(Train_Loader))
        if epoch %10==0:
            with torch.no_grad():
                sample_image = data[0].unsqueeze(0)  # Take the first image from the batch for display
                reconstructed_image, _, _ = vae(sample_image)

                # Create a grid of original and reconstructed images
                concatenated_image = vutils.make_grid(torch.cat((sample_image, reconstructed_image), dim=0), nrow=2, padding=2)
                
                # Save the concatenated image
                vutils.save_image(concatenated_image.cpu(), os.path.join(save_dir, f'concatenated_epoch_{epoch}.png'), normalize=True)
    torch.save(vae, 'trained_model_VAE/VAE.pth')
    with open("avg_loss.txt", 'w') as f:
        for s in avg_loss:
            f.write(str(s) + '\n')
