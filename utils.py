import torch
import torch.nn.functional as F
from PIL import Image

def combine_masks(garment_mask, head_mask, x, y, w, h):
    # Resize the head mask to match (w, h)
    head_mask = F.interpolate(head_mask.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
    head_mask = head_mask.squeeze(0)

    # Create an empty tensor for the combined mask
    combined_mask = torch.zeros_like(garment_mask)

    # Calculate the coordinates to place the head mask
    x_start, x_end = max(0, x), min(x + w, garment_mask.size(2))
    y_start, y_end = max(0, y), min(y + h, garment_mask.size(1))

    # Calculate the overlapping region
    x_head_start = max(0, -x)
    x_head_end = x_head_start + (x_end - x_start)
    y_head_start = max(0, -y)
    y_head_end = y_head_start + (y_end - y_start)

    # Copy the garment mask into the combined mask
    combined_mask[:, y_start:y_end, x_start:x_end] = garment_mask[:, y_start:y_end, x_start:x_end]

    # Overlay the resized head mask on the combined mask
    combined_mask[:, y_start:y_end, x_start:x_end] += head_mask[:, y_head_start:y_head_end, x_head_start:x_head_end]
    combined_mask = torch.clamp(combined_mask, 0, 1)

    return combined_mask
