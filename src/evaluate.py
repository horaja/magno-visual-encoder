# horaja/magno-visual-encoder/magno-visual-encoder-6421c0b14adc10a_deeper/src/evaluate.py

import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from model import SelectiveMagnoViT
from dataset import ImageNetteDataset # Make sure this matches your updated dataset script

def tensor_to_pil(tensor):
    """Convert a tensor image to a PIL Image for visualization."""
    # Un-normalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.clone().cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

def create_selection_mask(indices, num_patches_side, patch_size):
    """Create a mask image from the selected patch indices."""
    mask = torch.zeros(1, num_patches_side, num_patches_side)
    for idx in indices:
        row = idx // num_patches_side
        col = idx % num_patches_side
        mask[0, row, col] = 1
    
    # Upscale the mask to the full image size
    mask_upscaled = F.interpolate(
        mask.unsqueeze(0), 
        size=(num_patches_side * patch_size, num_patches_side * patch_size), 
        mode='nearest'
    )
    return mask_upscaled.squeeze()

def visualize_patch_selection(model, dataloader, device, output_path, num_samples=5):
    """
    Generates and saves a visualization of the patch selection process.
    """
    model.eval()
    
    # Get a single batch of data
    batch = next(iter(dataloader))
    color_images = batch['color_image'][:num_samples].to(device)
    line_drawings = batch['line_drawing'][:num_samples].to(device)

    # Get the selected indices from the model
    selected_indices = model.get_selected_indices(line_drawings)

    # Setup plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    fig.suptitle('Patch Selection Visualization', fontsize=16)

    patch_size = model.patch_size
    num_patches_side = model.img_size // patch_size

    for i in range(num_samples):
        # 1. Original Color Image
        ax = axes[i, 0]
        color_pil = tensor_to_pil(color_images[i])
        ax.imshow(color_pil)
        ax.set_title(f"Sample {i+1}: Original Color Image")
        ax.axis('off')

        # 2. Line Drawing "Map"
        ax = axes[i, 1]
        ld_resized = F.interpolate(line_drawings[i].unsqueeze(0), size=(model.img_size, model.img_size), mode='bilinear').squeeze()
        ax.imshow(ld_resized.cpu().numpy(), cmap='gray')
        ax.set_title("Line Drawing 'Map'")
        ax.axis('off')

        # 3. Selected Patches Visualization
        ax = axes[i, 2]
        mask = create_selection_mask(selected_indices[i], num_patches_side, patch_size)
        ax.imshow(color_pil)
        # Overlay a semi-transparent red mask
        ax.imshow(mask.cpu().numpy(), cmap='Reds', alpha=0.4)
        ax.set_title("Selected Patches")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transformations (should be for validation, no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and Dataloader
    val_dataset = ImageNetteDataset(
        color_root=args.color_dir,
        lines_root=args.lines_dir,
        split='val',
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Load model from checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Infer parameters from the checkpoint config if not provided
    if 'config' in checkpoint:
        config = checkpoint['config']
        args.img_size = config.get('img_size', args.img_size)
        args.patch_size = config.get('patch_size', args.patch_size)
        args.patch_percentage = config.get('patch_percentage', args.patch_percentage)
        num_classes = config.get('num_classes', 10) # Default to 10 if not in config
    else:
        # Fallback if no config in checkpoint
        num_classes = 10 # Adjust if necessary
        print("Warning: No config found in checkpoint. Using default parameters.")

    model = SelectiveMagnoViT(
        patch_percentage=args.patch_percentage,
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size
    ).to(device)

    # Adjust for different state_dict keys if necessary
    model_state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Generate the visualization
    visualize_patch_selection(model, val_loader, device, args.output_path, num_samples=args.num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SelectiveMagnoViT and visualize patch selection.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('--color_dir', type=str, required=True, help='Root directory of the 256x256 color images.')
    parser.add_argument('--lines_dir', type=str, required=True, help='Root directory of the 64x64 line drawings.')
    parser.add_argument('--output_path', type=str, default='results/plots/patch_selection_visualization.png', help='Path to save the output visualization.')
    
    parser.add_argument('--img_size', type=int, default=256, help='Size of the color images.')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of the patches.')
    parser.add_argument('--patch_percentage', type=float, default=0.4, help='Fraction of patches selected.')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize.')

    args = parser.parse_args()
    main(args)