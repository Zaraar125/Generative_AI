# Function to check if an image contains a person
# def is_person_image(image):
#     # For demonstration, assume this function returns True if a person is detected in the image
#     # You may use a pre-trained model like a face detector here
#     return True  # Replace with actual detection logic

# # Function to process the uploaded image and generate the result
# def generate_result(image):
#     if image is not None:
#         # Check if the image contains a person
#         if is_person_image(image):
#             # If it's a person, generate a sketch
#             result = sketch_generator(image)
#         else:
#             # Otherwise, generate a face image
#             result = face_generator(image)
#         return result
#     return None

# # Gradio interface
# with gr.Blocks() as demo:
#     # Image upload button
#     image_input = gr.Image(label="Upload Image", type="pil")

#     # Result display area
#     result_display = gr.Image(label="Result")

#     # Button to trigger the result generation
#     generate_btn = gr.Button("Generate Result")

#     # Link button to function
#     generate_btn.click(generate_result, inputs=image_input, outputs=result_display)

# # Launch the app
# demo.launch()