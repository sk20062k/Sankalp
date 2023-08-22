import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from dataloader import CustomDataset  # Import your custom DataLoader script
from generator import CVModelWithSkipConnections  # Import your custom model
# import wandb
from torch.optim import lr_scheduler

import cv2
import numpy as np


def bbox_mse_loss(bbox_pred, bbox_target):
    """
    Calculate the Mean Squared Error (MSE) loss between predicted and target bounding boxes.

    Args:
        bbox_pred (torch.Tensor): Predicted bounding boxes in the format (x, y, w, h).
        bbox_target (torch.Tensor): Target bounding boxes in the format (x, y, w, h).

    Returns:
        torch.Tensor: The MSE loss.
    """
    return F.mse_loss(bbox_pred.float(), bbox_target.float())


def combine_masks(x, y, w, h):
    
    batch_size = garment_mask.size(0)

    combined_masks = []
    BBoxs = []

    for i in range(batch_size):

        h_int = int(h[i].item())
        w_int = int(w[i].item())
        
        x_int = int(x[i].item())
        y_int = int(y[i].item())
        
        BBoxs.append(torch.tensor([x_int,y_int,w_int,h_int]))
        

    BBoxs = torch.stack(BBoxs)
    
    return BBoxs



# Define hyperparameters
batch_size = 4
num_epochs = 10
learning_rate = 0.01

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.01,
#     "architecture": "CNN",
#     "epochs": 10,
#     }
# )

# Specify data directory
data_dir = 'data'

print("data_dir", data_dir)

# Data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet normalization
])

# Create custom dataset and data loader
custom_dataset = CustomDataset(data_dir, transform=data_transforms)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Create model
model = CVModelWithSkipConnections()


# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for garment_mask, head_mask, true_mask,BBoxCord in dataloader:
        garment_mask, head_mask, true_mask,BBoxCord = garment_mask.to(device), head_mask.to(device), true_mask.to(device),BBoxCord.to(device)

        optimizer.zero_grad()
        predicted_values = model(garment_mask, head_mask)
        print(predicted_values.shape)
        BBoxPred = combine_masks(predicted_values[:, 0], predicted_values[:, 1], predicted_values[:, 2], predicted_values[:, 3])

        predicted_values=predicted_values.to(device)
        predicted_values = predicted_values.to(torch.float32)
        BBoxCord = BBoxCord.to(torch.float32)
       

        bbLoss = bbox_mse_loss(predicted_values,BBoxCord)
        
        bbLoss.backward()
        optimizer.step()
        
        print(predicted_values[0],BBoxCord[0])

        total_loss += bbLoss
        # wandb.log({
        #            "bbLoss":bbLoss})

    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(dataloader)}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')