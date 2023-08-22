import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2



class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.person_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    def __len__(self):
        return len(self.person_folders)
    
    def resize_image_and_bbox(self,original_image, original_bbox, new_size):
        # Get the original image size
        original_image_size = original_image.size

        # Calculate the resizing factor for both width and height
        resize_factor_width = new_size[0] / original_image_size[0]
        resize_factor_height = new_size[1] / original_image_size[1]

        # Calculate the new bounding box coordinates and dimensions
        new_bbox_x = int(original_bbox[0] * resize_factor_width)
        new_bbox_y = int(original_bbox[1] * resize_factor_height)
        new_bbox_width = int(original_bbox[2] * resize_factor_width)
        new_bbox_height = int(original_bbox[3] * resize_factor_height)

        return [new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height]

    def crop_head_mask_with_contour(self,head_mask):
        
        # Convert the PIL image to a NumPy array
        head_mask_np = np.array(head_mask)

        # Find contours in the head_mask image
        contours, _ = cv2.findContours(head_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the head_mask based on (x, y, w, h)
            head_mask_cropped = head_mask.crop((x, y, x + w, y + h))

            # Return the cropped head mask as a Pillow image with 'L' mode (grayscale)
            return head_mask_cropped,[x,y,w,h]
        # If no contours are found, return the original image
        return head_mask,[1,1,1,1]

    def __getitem__(self, idx):
        person_folder_name = self.person_folders[idx]
        person_folder_path = os.path.join(self.data_dir, person_folder_name)
        
        garment_mask = Image.open(os.path.join(person_folder_path, 'garmentMask.png')).convert('L')
        head_mask = Image.open(os.path.join(person_folder_path, 'headMask.png')).convert('L')
        head_mask_resized,BBoxOld = self.crop_head_mask_with_contour(head_mask)
        headCoord = self.resize_image_and_bbox(head_mask,BBoxOld,[256,256])
        complete_mask = Image.open(os.path.join(person_folder_path, 'completeMask.png')).convert('L')

        if self.transform:
            garment_mask = self.transform(garment_mask)
            head_mask_resized = self.transform(head_mask_resized)
            complete_mask = self.transform(complete_mask)
        
        return garment_mask, head_mask_resized, complete_mask,torch.tensor(headCoord)


# if __name__ == '__main__':
#     # Example of how to use the dataloader
#     data_dir = 'your_data_directory'
#     dataloader = get_custom_dataloader(data_dir, batch_size=32, shuffle=True)

#     for batch in dataloader:
#         garment_mask, head_mask, complete_mask = batch
#         # Perform training or other operations here
