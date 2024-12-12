import matplotlib.pyplot as plt
from SN_GAN import Generator
import torch
def generate_images(latent_dim, generator_path='trained_model_SN_GAN/generator.pth', num_images=16):
    """Generate and display multiple images from the GAN generator."""
    # Load the trained Generator model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = Generator(latent_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    
    
    # Generate random latent vectors
    z = torch.randn(num_images, latent_dim, 1, 1, device=device) # Batch size for multiple images
    
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

generate_images(256)

