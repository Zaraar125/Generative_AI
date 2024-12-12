import torch
import torch.nn as nn
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
# Generator
# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_channels):
#         super(Generator, self).__init__()
        
#         self.init_size = (64 // 4, 128 // 4)  # Initial size before upsampling (for a 64x128 image)
#         self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size[0] * self.init_size[1]))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1),
#             nn.Tanh()  # Output values will be between -1 and 1
#         )

#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size[0], self.init_size[1])  # Reshape to 128 channels
#         img = self.conv_blocks(out)
#         return img

# # Discriminator
# class Discriminator(nn.Module):
#     def __init__(self, img_channels):
#         super(Discriminator, self).__init__()

#         self.conv_blocks = nn.Sequential(
#             nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.fc_layer = nn.Sequential(
#             nn.Linear(128 * (64 // 8) * (128 // 8), 1),  # Flatten after the final conv block
#             nn.Sigmoid()  # Output a probability (real/fake)
#         )

#     def forward(self, img):
#         out = self.conv_blocks(img)
#         out = out.view(out.size(0), -1)  # Flatten the feature maps
#         validity = self.fc_layer(out)
#         return validity


# def Train_GAN(Train_Loader):
#     # Hyperparameters
#     latent_dim = 128  # Increase latent dimension
#     img_channels = 3
#     epochs = 600
#     lr_G = 0.0001
#     lr_D = 0.00001
#     phase_length = 10  # Number of epochs per phase

#     output_dir = 'GAN_Training_Outputs'
#     os.makedirs(output_dir, exist_ok=True)

#     # Initialize models
#     generator = Generator(latent_dim, img_channels).cuda()
#     discriminator = Discriminator(img_channels).cuda()

#     # Optimizers
#     optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

#     adversarial_loss = nn.BCELoss()
#     avg_loss_G=[]
#     avg_loss_D=[]
#     for epoch in range(epochs):
#         a=0
#         b=0
#         for i, (real_imgs) in enumerate(Train_Loader):
#             real_imgs = real_imgs.cuda()

#             # Train Discriminator
#             optimizer_D.zero_grad()

#             valid = torch.ones(real_imgs.size(0), 1).cuda() * 0.9
#             fake = torch.zeros(real_imgs.size(0), 1).cuda() * 0.1  # Softened fake labels

#             # Generate fake images
#             z = torch.randn(real_imgs.size(0), latent_dim).cuda()
#             gen_imgs = generator(z)

#             # Discriminator loss
#             real_loss = adversarial_loss(discriminator(real_imgs), valid)
#             fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
#             d_loss = (real_loss + fake_loss) / 2
#             d_loss.backward()
#             optimizer_D.step()

#             # Train Generator more frequently (every other step)
#             # if i % 2 == 0:
#             optimizer_G.zero_grad()
#             g_loss = adversarial_loss(discriminator(gen_imgs), valid)

#             if epoch>15:
#                 optimizer_G.zero_grad()
#                 g_loss = adversarial_loss(discriminator(gen_imgs), valid)
#                 g_loss.backward()
#                 optimizer_G.step()

#             # if i % 50 == 0:
#             #     print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(Train_Loader)}] [D Loss: {d_loss.item():.4f}] [G Loss: {g_loss.item():.4f}]")

#             a=a+d_loss.item()
#             b=b+g_loss.item()
#         avg_loss_D.append(a/len(Train_Loader))
#         avg_loss_G.append(b/len(Train_Loader))
#         print('[Epoch ',epoch,']',' [G LOSS ',b/len(Train_Loader),']  [D LOSS ',a/len(Train_Loader),']')
#         # Save real and generated images every 10 epochs
#         if (epoch + 1) % phase_length == 0:
#             concatenated_images = torch.cat((real_imgs[:8], gen_imgs[:8]), dim=0)
#             save_image(concatenated_images, f"{output_dir}/epoch_{epoch + 1}.png", nrow=4, normalize=True)
#     torch.save(generator,'New_trained_model_GAN/G.pth')
#     torch.save(discriminator,'New_trained_model_GAN/D.pth')
#     with open("New_avg_loss_D.txt", 'w') as f:
#         for s in avg_loss_D:
#             f.write(str(s) + '\n')
#     with open("New_avg_loss_G.txt", 'w') as f:
#         for s in avg_loss_G:
#            f.write(str(s) + '\n')

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        
        self.init_size = (64 // 4, 128 // 4)  # Initial size before upsampling (for a 64x128 image)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size[0] * self.init_size[1]))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output values will be between -1 and 1
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size[0], self.init_size[1])  # Reshape to 128 channels
        img = self.conv_blocks(out)
        return img

# # Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(128 * (64 // 8) * (128 // 8), 1),  # Flatten after the final conv block
            nn.Sigmoid()  # Output a probability (real/fake)
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)  # Flatten the feature maps
        validity = self.fc_layer(out)
        return validity


def Train_GAN(Train_Loader):
    # Hyperparameters
    latent_dim = 128  # Increase latent dimension
    img_channels = 3
    epochs = 600
    lr_G = 2e-4
    lr_D = 1e-5
    phase_length = 10  # Number of epochs per phase

    output_dir = 'GAN_Training_Outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    generator = Generator(latent_dim, img_channels).cuda()
    discriminator = Discriminator(img_channels).cuda()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss()
    avg_loss_G=[]
    avg_loss_D=[]
    for epoch in range(epochs):
        a=0
        b=0
        for i, (real_imgs) in enumerate(Train_Loader):
            real_imgs = real_imgs.cuda()

            # Train Discriminator
            optimizer_D.zero_grad()

            valid = torch.ones(real_imgs.size(0), 1).cuda() * 0.9
            fake = torch.zeros(real_imgs.size(0), 1).cuda() * 0.1  # Softened fake labels

            # Generate fake images
            z = torch.randn(real_imgs.size(0), latent_dim).cuda()
            gen_imgs = generator(z)

            # Discriminator loss
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator more frequently (every other step)
            if i % 2 == 0:
                optimizer_G.zero_grad()
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()
            a=a+d_loss.item()
            b=b+g_loss.item()
            # if i % 50 == 0:
            #     print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(Train_Loader)}] [D Loss: {d_loss.item():.4f}] [G Loss: {g_loss.item():.4f}]")
        avg_loss_D.append(a/len(Train_Loader))
        avg_loss_G.append(b/len(Train_Loader))
        print('[Epoch ',epoch,']',' [G LOSS ',b/len(Train_Loader),']  [D LOSS ',a/len(Train_Loader),']')
        # Save real and generated images every 10 epochs
        if (epoch + 1) % phase_length == 0:
            concatenated_images = torch.cat((real_imgs[:8], gen_imgs[:8]), dim=0)
            save_image(concatenated_images, f"{output_dir}/epoch_{epoch + 1}.png", nrow=8, normalize=True)
    torch.save(generator,'trained_model_GAN/G.pth')
    torch.save(discriminator,'trained_model_GAN/D.pth')
    with open("avg_loss_D.txt", 'w') as f:
        for s in avg_loss_D:
            f.write(str(s) + '\n')
    with open("avg_loss_G.txt", 'w') as f:
        for s in avg_loss_G:
           f.write(str(s) + '\n')