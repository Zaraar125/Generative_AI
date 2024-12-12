import torch
import warnings
from torchvision.utils import save_image
warnings.filterwarnings('ignore')
import torch
import matplotlib.pyplot as plt

def generate_images(latent_dim, generator_path='trained_model_GAN/G.pth', num_images=16):
    """Generate and display multiple images from the GAN generator."""
    # Load the trained Generator model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = torch.load(generator_path).to(device)
    generator.eval()  # Set to evaluation mode
    
    # Generate random latent vectors
    z = torch.randn(num_images, latent_dim).to(device)  # Batch size for multiple images
    
    # Generate the images
    with torch.no_grad():  # No gradient calculation for inference
        generated_imgs = generator(z)

    # Denormalize the images
    generated_imgs = (generated_imgs + 1) / 2  # Scale to [0, 1] range from [-1, 1]

    # Plotting the images
    plt.figure(figsize=(10, 10))
    
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)  # Create a 4x4 grid
        plt.imshow(generated_imgs[i].cpu().numpy().transpose(1, 2, 0))  # Convert to HWC format
        plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

def infer_and_show_generated_images(model_path, num_images=16, latent_dim=64, device='cuda'):
    """Load the trained VAE model, generate images, and display them."""
    # Load the trained VAE model
    vae = torch.load(model_path)
    vae.eval()  # Set the model to evaluation mode

    # Sample random points from the latent space
    z = torch.randn(num_images, latent_dim).to(device)  # Latent vector

    # Generate images using the decoder
    with torch.no_grad():
        generated_images = vae.decode(z)

    # Show generated images
    generated_images = generated_images.cpu().detach().numpy()
    
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].transpose(1, 2, 0))  # Change shape for imshow
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Example usage
# generate_images(latent_dim=128)  # Adjust latent_dim according to your model

# Example usage
infer_and_show_generated_images('trained_model_VAE/VAE.pth')


