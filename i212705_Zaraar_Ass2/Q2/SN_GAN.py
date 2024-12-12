import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
device='cuda'
# Define the Generator Model
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),  # Output: 256 x 4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # Output: 128 x 8 x 8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),     # Output: 64 x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),       # Output: 3 x 32 x 32 (Correct size)
            nn.Tanh()  # Output values between [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

# Define the Similarity Network (Discriminator)
class SimilarityNetwork(nn.Module):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 64),  # Flattened output from 8x8 feature map
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output similarity score between [0, 1]
        )

    def forward(self, img1, img2):
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)

        # Flatten the feature maps
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        # Calculate L1 distance between features
        distance = torch.abs(features1 - features2)

        # Compute similarity score
        similarity_score = self.fc(distance)
        return similarity_score

def Train_SN_GAN(train_loader):
        # Parameters
        z_dim = 256  # Size of latent vector
        num_epochs = 30
        lr_G = 2e-4  # Learning rate for Generator
        lr_D = 1e-5  # Learning rate for Discriminator
        beta1 = 0.5

        # Initialize the models
        netG = Generator(z_dim).to(device)
        netD = SimilarityNetwork().to(device)

        # Loss function and optimizers
        criterion = nn.BCELoss()  # Binary cross entropy loss
        optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
        avg_loss_D=[]
        avg_loss_G=[]
        # Training loop
        for epoch in range(num_epochs):
            a=0
            b=0
            for i, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(device)  # Shape: [batch_size, channels, height, width]
                b_size = real_images.size(0)

                # Generate fake images
                noise = torch.randn(b_size, z_dim, 1, 1, device=device)
                fake_images = netG(noise)

                # Reset gradients
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                # Discriminator training
                # Calculate similarity for fake images
                similarity_fake = netD(fake_images.detach(), real_images)  # Compare fake images with real images
                fake_labels = torch.zeros(b_size, device=device)  # Fake images should be labeled as dissimilar

                # Compute loss for the discriminator
                similarity_fake=similarity_fake.squeeze(-1)
                lossD = criterion(similarity_fake, fake_labels)  # Discriminator should maximize dissimilarity
                lossD.backward()
                optimizerD.step()

                # Calculate similarity for the generator
                similarity_gen = netD(fake_images, real_images)  # Compare fake images with real images
                real_labels = torch.ones(b_size, device=device)*0.9  # Generator should be labeled as similar

                # Compute loss for the generator
                similarity_gen=similarity_gen.squeeze(-1)
                lossG = criterion(similarity_gen, real_labels)  # Generator should minimize the similarity score
                lossG.backward()  # Only backpropagate for the generator
                optimizerG.step()
                a=a+lossD.item()
                b=b+lossG.item()
                if i % 100 == 0:

                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                            f'Generator Loss: {lossG.item():.4f}, Discriminator Loss: {lossD.item():.4f}')
            avg_loss_D.append(a)
            avg_loss_G.append(b)
            # Save fake images for visualization every 5 epochs
            if epoch % 5 == 0:
                vutils.save_image(fake_images.detach(), f'SN_GAN_Training_Output/fake_samples_epoch_{epoch}.png', normalize=True)
        torch.save(netG.state_dict(), 'generator.pth')
        torch.save(netD.state_dict(), 'siamese_discriminator.pth')
        with open("avg_loss_D.txt", 'w') as f:
            for s in avg_loss_D:
                f.write(str(s) + '\n')
        with open("avg_loss_G.txt", 'w') as f:
            for s in avg_loss_G:
                f.write(str(s) + '\n')

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import torchvision.utils as vutils
# device='cuda'
# # Define the Generator Model
# class Generator(nn.Module):
#     def __init__(self, z_dim):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),  # Output: 256 x 4 x 4
#             nn.ReLU(True),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # Output: 128 x 8 x 8
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),     # Output: 64 x 16 x 16
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),       # Output: 3 x 32 x 32 (Correct size)
#             nn.Tanh()  # Output values between [-1, 1]
#         )

#     def forward(self, input):
#         return self.main(input)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 2, 1),  # 32x32 -> 16x16
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1),  # 16x16 -> 8x8
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1),  # 8x8 -> 4x4
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 1, 4, 1, 0),  # 4x4 -> 1x1
#             nn.Sigmoid()  # Output probability between [0, 1]
#         )

#     def forward(self, input):
#         return self.main(input).view(-1)

# def Train_SN_GAN(train_loader):
#     # Parameters
#     z_dim = 256  # Size of latent vector
#     num_epochs = 120
#     lr_G = 2e-4  # Learning rate for Generator
#     lr_D = 1e-5  # Learning rate for Discriminator
#     beta1 = 0.5

#     # Initialize the models
#     netG = Generator(z_dim).to(device)
#     netD = Discriminator().to(device)

#     # Loss function and optimizers
#     criterion = nn.BCELoss()  # Binary cross-entropy loss
#     optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
#     optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

#     # Training loop
#     avg_loss_D=[]
#     avg_loss_G=[]
#     for epoch in range(num_epochs):
#         a=0
#         b=0
#         for i, (real_images, _) in enumerate(train_loader):
#             real_images = real_images.to(device)  # Shape: [batch_size, channels, height, width]
#             b_size = real_images.size(0)

#             # Generate fake images
#             noise = torch.randn(b_size, z_dim, 1, 1, device=device)
#             fake_images = netG(noise)

#             # Reset gradients
#             optimizerD.zero_grad()

#             # Train Discriminator
#             # Real images
#             real_labels = torch.ones(b_size, device=device) * 0.9  # Smooth real labels for stability
#             output_real = netD(real_images)
#             lossD_real = criterion(output_real, real_labels)

#             # Fake images
#             fake_labels = torch.zeros(b_size, device=device)
#             output_fake = netD(fake_images.detach())
#             lossD_fake = criterion(output_fake, fake_labels)

#             # Total loss and update for Discriminator
#             lossD = lossD_real + lossD_fake
#             lossD.backward()
#             optimizerD.step()

#             # Train Generator
#             optimizerG.zero_grad()
#             # Try to fool the discriminator with fake images as real (label = 1)
#             output_fake = netD(fake_images)
#             lossG = criterion(output_fake, real_labels)
#             lossG.backward()
#             optimizerG.step()

#             a=a+lossD.item()
#             b=b+lossG.item()
#             if i % 100 == 0:
#                 print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
#                       f'Generator Loss: {lossG.item():.4f}, Discriminator Loss: {lossD.item():.4f}')
#         avg_loss_D.append(a)
#         avg_loss_G.append(b)
#         # Save fake images for visualization every 5 epochs
#         if epoch % 5 == 0:
#                 real_images = real_images[:8]
#                 fake_images = fake_images[:8]

#                 # Concatenate the images along the batch dimension (0) to create a two-row grid
#                 images_to_save = torch.cat([real_images, fake_images], dim=0)  # Shape: [16, C, H, W]

#                 # Create a grid with 8 images per row
#                 grid = vutils.make_grid(images_to_save, nrow=8, normalize=True)

#                 # Save the image grid
#                 vutils.save_image(grid, f'SN_GAN_Training_Output/real_and_fake_epoch_{epoch}.png', normalize=True)

#             # vutils.save_image(fake_images.detach(), f'SN_GAN_Training_Output/fake_samples_epoch_{epoch}.png', normalize=True)
#     torch.save(netG.state_dict(), 'trained_model_SN_GAN/generator.pth')
#     torch.save(netD.state_dict(), 'trained_model_SN_GAN/discriminator.pth')
#     with open("avg_loss_D.txt", 'w') as f:
#         for s in avg_loss_D:
#             f.write(str(s) + '\n')
#     with open("avg_loss_G.txt", 'w') as f:
#         for s in avg_loss_G:
#            f.write(str(s) + '\n')
