import torch
from torchvision import transforms
from PIL import Image
from generator import CVModelWithSkipConnections  # Import your custom model
import cv2
import numpy as np

def preprocess_and_predict(garment_path, face_path, model_path):
    # Load the model
    model = CVModelWithSkipConnections()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    def preprocess_image(image_path):
        # Load the image
        image = Image.open(image_path)

        # Convert the image to RGB mode (if it's not already)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get the image dimensions
        width, height = image.size

        # Create a new blank image in L mode (grayscale)
        new_image = Image.new("L", (width, height), color=255)

        # Iterate over each pixel in the original RGB image
        for x in range(width):
            for y in range(height):
                r, g, b = image.getpixel((x, y))

                # Check if the pixel is not black (all three channels are not 0)
                if r != 0 or g != 0 or b != 0:
                    # Set the corresponding pixel in the new image to black (0)
                    new_image.putpixel((x, y), 0)

        return new_image, width, height

    # Define data transformation for the input images
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet normalization
    ])

    # Load and preprocess the first test image
    garment, width, height = preprocess_image(garment_path)
    garment = data_transform(garment).unsqueeze(0)  # Add a batch dimension

    # Load and preprocess the second test image
    face, _, _ = preprocess_image(face_path)
    face = data_transform(face).unsqueeze(0)  # Add a batch dimension

    # Make predictions for the test images
    with torch.no_grad():
        predicted_values = model(garment, face)  # Assuming garment_mask and head_mask are the same for both images

    # Convert predicted bounding box values to integers and extract them
    predicted_bbox = predicted_values[0].cpu().numpy().astype(int)

    # Calculate the scaling factors
    scale_x = width / 256.0
    scale_y = height / 256.0

    # Scale the predicted bounding box
    scaled_bbox = predicted_bbox.copy()  # Make a copy to avoid modifying the original

    # Scale the x and w coordinates
    scaled_bbox[0] = int(scaled_bbox[0] * scale_x)
    scaled_bbox[2] = int(scaled_bbox[2] * scale_x)

    # Scale the y and h coordinates
    scaled_bbox[1] = int(scaled_bbox[1] * scale_y)
    scaled_bbox[3] = int(scaled_bbox[3] * scale_y)

    return scaled_bbox

# Example usage:
garment_path = 'path_to_image1.jpg'
face_path = 'path_to_image2.jpg'
model_path = 'trained_model.pth'

scaled_bbox = preprocess_and_predict(garment_path, face_path, model_path)
print("Scaled Bounding Box for Image 1 (x, y, w, h):", scaled_bbox)
