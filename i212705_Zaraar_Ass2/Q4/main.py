import gradio as gr
from PIL import Image
from Cycle_GAN import Generator
import torch
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the generators (Assuming Generator is a model for both sketches and faces)
sketch_generator = Generator(img_channels=3, num_residuals=9).to(device)  # For generating sketches
face_generator = Generator(img_channels=3, num_residuals=9).to(device)    # For generating faces

# Load pre-trained weights if available
checkpoint_sketch = torch.load("genS.pth.tar", map_location=device)
checkpoint_face = torch.load("genP.pth.tar", map_location=device)

# Access only the model weights within the checkpoint
sketch_generator.load_state_dict(checkpoint_sketch['state_dict'])
face_generator.load_state_dict(checkpoint_face['state_dict'])

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming models were trained with this normalization
])

# Function to process the uploaded image and generate the result
def generate_result(image, mode):
    if image is not None:
        # Preprocess the image
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Check the selected mode to decide which generator to use
        if mode == "Sketch":
            # Generate a sketch from the input image
            result = face_generator(image)
        elif mode == "Face":
            # Generate a face from the input image
            result = sketch_generator(image)
        else:
            return None

        # Convert the output back to a PIL image
        result = result.squeeze(0).cpu().detach()
        result = transforms.ToPILImage()(result)
        return result
    return None

# Gradio interface
with gr.Blocks() as demo:
    # Image upload button
    image_input = gr.Image(label="Upload Image", type="pil")

    # Dropdown to select mode (Sketch or Face generation)
    mode_input = gr.Dropdown(choices=["Sketch", "Face"], label="Select Mode")

    # Result display area
    result_display = gr.Image(label="Result")

    # Button to trigger the result generation
    generate_btn = gr.Button("Generate Result")

    # Link button to function
    generate_btn.click(generate_result, inputs=[image_input, mode_input], outputs=result_display)

# Launch the app
demo.launch()
