
import os
import cv2
import torch 
import numpy as np
import pandas as pd

def read_images_and_store_in_directory(image_paths,saving_directory):
    # # Load the image
    z = cv2.imread(r'i212705_Zaraar_Ass1\Question_1\Data\20240904_083201.jpg')
    # z=cv2.imread(r'i212705_Zaraar_Ass1\Question_1\Data\20240904_091451.jpg')
    image=cv2.resize(z,(2000,1080))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use binary thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Use edge detection to detect edges in the image
    edges = cv2.Canny(thresh, 100, 200)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Define the number of boxes and rectangles
    num_boxes = 10
    num_rectangles = 10

    # Calculate the height of each box (for square boxes)
    total_height = y 
    box_height = total_height // num_boxes  

    rectangle_width = 2*w+120
    rectangle_height = box_height 

    # Copy the original image for drawing
    image_with_shapes = image.copy()

    # Create a directory to save the boxes (optional)
    # output_dir = 'segmented_boxes_and_rectangles'
    # os.makedirs(output_dir, exist_ok=True)

    # Draw 10 square boxes above the largest contour
    for i in range(num_boxes):
        # Calculate the top and bottom coordinates for the current box
        top_y = y - (i + 1) * box_height
        bottom_y = y - i * box_height
        
        # Draw the square boxes on the image
        cv2.rectangle(image_with_shapes, (x, top_y), (x + w, bottom_y), (255, 0, 0), 2)

        box_region = image[top_y:bottom_y, x:x + w]
        # Draw 10 square boxes above the largest contour
    for i in range(2):
        # Calculate the top and bottom coordinates for the current box
        top_y = y + (i + 1) * box_height
        bottom_y = y + i * box_height
        
        # Draw the square boxes on the image
        cv2.rectangle(image_with_shapes, (x, top_y), (x + w, bottom_y), (255, 0, 0), 2)

        box_region = image[top_y:bottom_y, x:x + w]

    # Draw 10 rectangles alongside the square boxes
    t=rectangle_width*4
    holder=0
    for i in range(18):
        for i in range(num_rectangles):
            # Calculate the top and bottom coordinates for the current rectangle
            top_y = y - (i + 1) * rectangle_height
            bottom_y = y - i * rectangle_height
            
            # Draw the rectangles next to the square boxes (positioned next to the square boxes)
            cv2.rectangle(image_with_shapes, (x + w + 10, top_y), (x + w + 10 + t, bottom_y), (0, 255, 0), 2)

            # Optionally crop and save each rectangle region as a separate image
            rectangle_region = image[top_y:bottom_y, x + w + 10:x + w + 10 + t]
            
            cv2.imwrite(os.path.join(saving_directory, f'rectangle_{i+1+holder}.png'), rectangle_region)
        # t=t+rectangle_width
        holder=holder+10
    # zzzz=170
    # for i in range(2):
    #     for i in range(11,13):
    #         # Calculate the top and bottom coordinates for the current rectangle
    #         top_y = y + (i + 1) * rectangle_height
    #         bottom_y = y + i * rectangle_height
            
    #         # Draw the rectangles next to the square boxes (positioned next to the square boxes)
    #         cv2.rectangle(image_with_shapes, (x + w + 10, top_y), (x + w + 10 + t, bottom_y), (0, 255, 0), 2)

    #         # Optionally crop and save each rectangle region as a separate image
    #         rectangle_region = image[top_y:bottom_y, x + w + 10:x + w + 10 + t]


    #         cv2.imwrite(os.path.join(saving_directory, f'rectangle_{i+1+zzzz}.png'), rectangle_region)
    
    cv2.imshow('Image with 10 Square Boxes and Rectangles', image_with_shapes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

