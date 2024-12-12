
import os
import shutil
import pandas as pd
from PIL import Image, ImageOps

def train_test_frames(root_dir):
    """
    Splits the dataset into training and testing dataframes for VAE training.
    
    Args:
        root_dir (string): Directory with all the images, where each subfolder represents a class.
    
    Returns:
        train_df (DataFrame): DataFrame with training image paths.
        test_df (DataFrame): DataFrame with testing image paths.
    """
    train_paths = []
    test_paths = []

    # Iterate through each subfolder in the root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            images = sorted(os.listdir(folder_path))  # Sort to maintain order of images

            # Get the first 3 images for training
            train_images = images[:3]
            # Get the last image for testing
            test_image = images[3]

            # Append training image paths
            for img_name in train_images:
                img_path = os.path.join(folder_path, img_name)
                train_paths.append(img_path)

            # Append testing image path
            test_img_path = os.path.join(folder_path, test_image)
            test_paths.append(test_img_path)

    # Create DataFrames (no labels needed for VAE)
    train_df = pd.DataFrame({
        'image_path': train_paths
    })

    test_df = pd.DataFrame({
        'image_path': test_paths
    })

    return train_df, test_df
def upscale_image(image, size=(224, 224)):
    return image
    img_w, img_h = image.size

    # If the image is smaller than the target size, upscale it
    if img_w < size[0] or img_h < size[1]:
        scale = min(size[0] / img_h, size[1] / img_w)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Pad the image to make it exactly 224x224
    new_image = ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    return new_image

def upscale_images(signatures_folder,output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each folder inside "Signatures"
    for folder_name in os.listdir(signatures_folder):
        folder_path = os.path.join(signatures_folder, folder_name)
        
        # Check if it is a folder
        if os.path.isdir(folder_path):
            images = os.listdir(folder_path)
            
            # Create a corresponding subfolder in the output folder
            output_subfolder = os.path.join(output_folder, folder_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            
            # Loop through the images and rename/upscale them
            for idx, image_name in enumerate(images):
                image_path = os.path.join(folder_path, image_name)
                
                # Open image
                img = Image.open(image_path)
                
                # Upscale the image to (224, 224)
                resized_img = upscale_image(img, size=(224, 224))
                
                # Rename the image according to the folder name
                new_image_name = f"{folder_name}_Image_{idx+1}.png"
                output_image_path = os.path.join(output_subfolder, new_image_name)
                
                # Save the image with the new name in the respective subfolder
                resized_img.save(output_image_path)