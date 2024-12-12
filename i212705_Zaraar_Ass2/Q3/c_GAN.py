import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# # Generator Model
# class Generator(nn.Module):
#     def __init__(self, input_channels=3, output_channels=3, ngf=64):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(ngf, output_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.main(x)

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)  # Skip connection (residual addition)

# Modified Generator Model with Residual Blocks
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, ngf=64, num_residual_blocks=3):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        # Add Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(num_residual_blocks)]
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),         
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),     
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0),           
        )
        self.fc = nn.Linear(625, 1)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.main(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return self.sigmoid(x)  




from tqdm import tqdm

def train_CGAN(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    G = Generator().to(device)
    D = Discriminator().to(device)
    num_epochs=50
    # Loss function
    criterion = nn.BCELoss()
    lr_G = 2e-4
    lr_D = 1e-5
    # Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
    avg_loss_D=[]
    avg_loss_G=[]
    for epoch in range(num_epochs):
        a=0
        b=0
        for i, (colored_images,sketches) in enumerate(tqdm(train_loader,ncols=100)):
            colored_images=colored_images.permute(0, 3, 1, 2)
            sketches=sketches.permute(0, 3, 1, 2)

            # Move data to device
            sketches = sketches.to(device)
            colored_images = colored_images.to(device)
            
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            batch_size = sketches.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)*0.9
            fake_labels = torch.zeros(batch_size, 1, device=device)*0.1
            
            real_labels=real_labels.squeeze(-1)
            fake_labels=fake_labels.squeeze(-1)
            
            fake_person_image=G(sketches)

            # Discriminator loss
            real_loss = criterion(D(colored_images).squeeze(-1), real_labels)
            fake_loss = criterion(D(fake_person_image.detach()).squeeze(-1), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            
            # Generator Loss
            if epoch%3==0:
                g_loss = criterion(D(fake_person_image).squeeze(-1), real_labels)
                g_loss.backward()
                optimizer_G.step()
            
            a=a+d_loss.item()
            b=b+g_loss.item()
        
            if i % 124 == 0:
                combined_images = torch.cat((colored_images, fake_person_image), dim=0)
                grid = vutils.make_grid(combined_images, nrow=8, padding=2, normalize=True)

                save_path = f"Q3/CGAN_training_images/epoch_{epoch}.png"
                vutils.save_image(grid, save_path)
        avg_loss_D.append(a/len(train_loader))
        avg_loss_G.append(b/len(train_loader))

        print('Epoch : ',epoch,' D Loss : ',a/len(train_loader),' G Loss : ',b/len(train_loader))
    torch.save(G.state_dict(), 'Q3/generator.pth')
    torch.save(D.state_dict(), 'Q3/discriminator.pth')
    with open("Q3/avg_loss_D.txt", 'w') as f:
        for s in avg_loss_D:
            f.write(str(s) + '\n')
    with open("Q3/avg_loss_G.txt", 'w') as f:
        for s in avg_loss_G:
            f.write(str(s) + '\n')
# Assuming `dataloader` provides (sketch, colored_image) pairs.
