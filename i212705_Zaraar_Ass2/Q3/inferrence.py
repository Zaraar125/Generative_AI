import matplotlib.pyplot as plt
import torch
from c_GAN import Generator
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Define the inference function
def generate_image_from_sketch(generator_path, sketch_image_path, output_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained generator model
    G = Generator().to(device)
    G.load_state_dict(torch.load(generator_path, map_location=device))
    G.eval()

    # Define the transformations for the input sketch image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalizing with mean and std dev as [0.5] for each channel
    ])

    # Load and preprocess the sketch image
    sketch_image = Image.open(sketch_image_path).convert("RGB")
    sketch_image = transform(sketch_image).unsqueeze(0).to(device)  # Add batch dimension

    # Generate the image from the sketch
    with torch.no_grad():
        generated_image = G(sketch_image)
    
    # Denormalize and save the generated image
    generated_image = generated_image.squeeze(0).cpu().detach()
    generated_image = (generated_image + 1) / 2  # Convert from [-1,1] to [0,1] range
    save_image(generated_image, output_image_path)
    
# Example usage
generate_image_from_sketch(
    generator_path="Q3/generator.pth",
    sketch_image_path=r"Q3\dataset\train\sketches\0.jpg",
    output_image_path="Q3/generated_image.png"
)

