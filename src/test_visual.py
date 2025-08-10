import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from fvcore.nn import FlopCountAnalysis

# Import our custom modules
from model import SelectiveMagnoViT
from dataset import ImageNetteDataset
from torchvision import transforms

def plot_confusion_matrix(cm, class_names, output_dir, patch_percentage):
    """Saves a confusion matrix plot to a file."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Patch Percentage: {patch_percentage*100}%)')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"confusion_matrix_pp{patch_percentage}.png")
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    
def visualize_patch_selection(model, val_loader, device, output_dir, num_samples=16, patch_percentage=None):
    """
    Visualize which patches are being selected by the model.
    
    Args:
        model: The SelectiveMagnoViT model
        val_loader: Validation dataloader
        device: CUDA or CPU device
        output_dir: Directory to save visualization
        num_samples: Number of samples to visualize
        patch_percentage: For filename purposes
    """
    model.eval()
    
    # Collect samples
    magno_images = []
    line_drawings = []
    selected_patches_list = []
    patch_scores_list = []
    predictions = []
    true_labels = []
    class_names = val_loader.dataset.class_names
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            magno_img = batch['magno_image'].to(device)
            line_img = batch['line_drawing'].to(device)
            label = batch['label'].to(device)
            
            # Get patch scores from the model
            patch_scores = model.scorer(line_img)
            
            # Get selected patch indices
            k = int(patch_scores.shape[1] * model.selector.patch_percentage)
            _, top_k_indices = torch.topk(patch_scores, k=k, dim=1)
            
            # Get prediction
            output = model(magno_img, line_img)
            _, pred = torch.max(output, 1)
            
            # Store for visualization
            magno_images.append(magno_img[0].cpu())
            line_drawings.append(line_img[0].cpu())
            selected_patches_list.append(top_k_indices[0].cpu())
            patch_scores_list.append(patch_scores[0].cpu())
            predictions.append(pred[0].cpu().item())
            true_labels.append(label[0].cpu().item())
    
    # Create visualization
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = GridSpec(num_samples, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx in range(min(num_samples, len(magno_images))):
        # 1. Original Magno Image
        ax1 = fig.add_subplot(gs[idx, 0])
        magno_np = denormalize_image(magno_images[idx])
        ax1.imshow(magno_np)
        ax1.set_title(f'Magno Image\nTrue: {class_names[true_labels[idx]]}', fontsize=10)
        ax1.axis('off')
        
        # 2. Line Drawing
        ax2 = fig.add_subplot(gs[idx, 1])
        line_np = line_drawings[idx].squeeze().numpy()
        ax2.imshow(line_np, cmap='gray')
        ax2.set_title(f'Line Drawing\nPred: {class_names[predictions[idx]]}', fontsize=10)
        ax2.axis('off')
        
        # 3. Selected Patches Mask (Blackout version)
        ax3 = fig.add_subplot(gs[idx, 2])
        masked_image = create_masked_image(
            magno_images[idx], 
            selected_patches_list[idx], 
            model.patch_size,
            blackout=True
        )
        ax3.imshow(masked_image)
        ax3.set_title(f'Selected Patches\n({k}/{patch_scores.shape[1]} patches)', fontsize=10)
        ax3.axis('off')
        
        # 4. Patch Grid Overlay
        ax4 = fig.add_subplot(gs[idx, 3])
        overlay_image = create_patch_overlay(
            magno_images[idx],
            selected_patches_list[idx],
            model.patch_size
        )
        ax4.imshow(overlay_image)
        ax4.set_title('Patch Grid Overlay', fontsize=10)
        ax4.axis('off')
        
        # 5. Patch Importance Heatmap
        ax5 = fig.add_subplot(gs[idx, 4])
        heatmap = create_importance_heatmap(
            patch_scores_list[idx],
            selected_patches_list[idx],
            model.patch_size,
            magno_images[idx].shape[-1]
        )
        ax5.imshow(heatmap, cmap='hot', interpolation='nearest')
        ax5.set_title('Importance Scores', fontsize=10)
        ax5.axis('off')
    
    # Save figure
    output_path = os.path.join(output_dir, f'patch_selection_visualization_pp{patch_percentage}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Patch visualization saved to {output_path}")

def denormalize_image(tensor):
    """Denormalize from ImageNet stats back to [0,1]"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

def create_masked_image(image_tensor, selected_indices, patch_size, blackout=True):
    """
    Create a masked version of the image showing only selected patches.
    
    Args:
        image_tensor: Original image tensor (C, H, W)
        selected_indices: Indices of selected patches
        patch_size: Size of each patch
        blackout: If True, black out unselected patches. If False, dim them.
    """
    img = denormalize_image(image_tensor.clone())
    H, W = img.shape[:2]
    patches_per_row = W // patch_size
    
    # Create mask
    mask = np.zeros((H, W))
    for idx in selected_indices:
        row = idx // patches_per_row
        col = idx % patches_per_row
        y_start = row * patch_size
        x_start = col * patch_size
        mask[y_start:y_start + patch_size, x_start:x_start + patch_size] = 1
    
    # Apply mask
    if blackout:
        img = img * mask[:, :, np.newaxis]
    else:
        # Dim unselected patches instead of blacking out
        img = img * (0.3 + 0.7 * mask[:, :, np.newaxis])
    
    return img

def create_patch_overlay(image_tensor, selected_indices, patch_size):
    """
    Create an overlay showing patch boundaries with selected patches highlighted.
    """
    img = denormalize_image(image_tensor.clone())
    H, W = img.shape[:2]
    patches_per_row = W // patch_size
    
    # Create figure for overlay
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    
    # Draw all patch boundaries
    for i in range(0, H, patch_size):
        ax.axhline(y=i, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(0, W, patch_size):
        ax.axvline(x=j, color='gray', linewidth=0.5, alpha=0.5)
    
    # Highlight selected patches
    for idx in selected_indices:
        row = idx // patches_per_row
        col = idx % patches_per_row
        rect = patches.Rectangle(
            (col * patch_size, row * patch_size),
            patch_size, patch_size,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.axis('off')
    
    # Convert to image array (FIXED)
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    overlay = np.asarray(buf)
    plt.close(fig)
    
    # Convert RGBA to RGB
    overlay = overlay[:, :, :3]
    
    # Resize to match original
    from PIL import Image
    overlay_pil = Image.fromarray(overlay)
    overlay_pil = overlay_pil.resize((W, H), Image.BILINEAR)
    
    return np.array(overlay_pil) / 255.0

def create_importance_heatmap(patch_scores, selected_indices, patch_size, image_size):
    """
    Create a heatmap showing the importance scores of each patch.
    """
    num_patches_per_dim = image_size // patch_size
    heatmap = patch_scores.reshape(num_patches_per_dim, num_patches_per_dim).numpy()
    
    # Upscale for better visualization
    heatmap_large = np.repeat(np.repeat(heatmap, patch_size, axis=0), patch_size, axis=1)
    
    # Mark selected patches
    mask = np.zeros_like(patch_scores)
    mask[selected_indices] = 1
    mask = mask.reshape(num_patches_per_dim, num_patches_per_dim)
    mask_large = np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1)
    
    # Add border to selected patches in heatmap
    heatmap_large = heatmap_large * (0.7 + 0.3 * mask_large)
    
    return heatmap_large
    
def analyze_patch_selection_statistics(model, val_loader, device, output_dir, patch_percentage):
    """
    Analyze statistics about which patches are most frequently selected.
    """
    model.eval()
    
    # Get image dimensions from first batch
    first_batch = next(iter(val_loader))
    img_size = first_batch['magno_image'].shape[-1]
    num_patches = (img_size // model.patch_size) ** 2
    
    # Track selection frequency
    patch_selection_counts = torch.zeros(num_patches)
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            line_drawings = batch['line_drawing'].to(device)
            patch_scores = model.scorer(line_drawings)
            
            k = int(patch_scores.shape[1] * model.selector.patch_percentage)
            _, top_k_indices = torch.topk(patch_scores, k=k, dim=1)
            
            # Count selections
            for indices in top_k_indices:
                for idx in indices:
                    patch_selection_counts[idx] += 1
            
            total_samples += line_drawings.shape[0]
    
    # Normalize to get frequencies
    patch_frequencies = patch_selection_counts / total_samples
    
    # Create visualization
    patches_per_dim = int(np.sqrt(num_patches))
    freq_grid = patch_frequencies.reshape(patches_per_dim, patches_per_dim).numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap of selection frequencies
    im = ax1.imshow(freq_grid, cmap='viridis', interpolation='nearest')
    ax1.set_title(f'Patch Selection Frequency\n(pp={patch_percentage})')
    ax1.set_xlabel('Patch Column')
    ax1.set_ylabel('Patch Row')
    plt.colorbar(im, ax=ax1)
    
    # Histogram of frequencies
    ax2.hist(patch_frequencies.numpy(), bins=30, edgecolor='black')
    ax2.set_xlabel('Selection Frequency')
    ax2.set_ylabel('Number of Patches')
    ax2.set_title('Distribution of Patch Selection Frequencies')
    ax2.axvline(patch_frequencies.mean(), color='red', linestyle='--', label=f'Mean: {patch_frequencies.mean():.3f}')
    ax2.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'patch_selection_statistics_pp{patch_percentage}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Patch selection statistics saved to {output_path}")
    print(f"Most selected patch: {patch_frequencies.argmax().item()} (freq: {patch_frequencies.max():.3f})")
    print(f"Least selected patch: {patch_frequencies.argmin().item()} (freq: {patch_frequencies.min():.3f})")
    print(f"Mean selection frequency: {patch_frequencies.mean():.3f}")
    print(f"Std of selection frequency: {patch_frequencies.std():.3f}")

def main(args):
    """
    Main function to evaluate the trained model.
    """
    # --- 1. Device and Directory Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.results_dir, exist_ok=True)

    # --- 2. Data Loading ---
    vit_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading validation data for testing...")
    val_dataset = ImageNetteDataset(
        magno_root=args.magno_dir,
        lines_root=args.lines_dir,
        split='val',
        transform=vit_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, # Use a batch size of 1 for latency/FLOPs analysis
        shuffle=False,
        num_workers=args.num_workers
    )
    
    num_classes = len(val_dataset.class_names)
    print(f"Number of classes: {num_classes}")

    # --- 3. Model Loading ---
    print("Initializing and loading model...")
    model = SelectiveMagnoViT(
        patch_percentage=args.patch_percentage,
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size
    ).to(device)

    model_path = os.path.join(args.model_dir, f"best_model_pp{args.patch_percentage}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded successfully from {model_path}")

    # --- 4. Evaluation ---
    all_preds = []
    all_labels = []
    total_latency = 0.0

    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating")
        for i, batch in enumerate(progress_bar):
            magno_images = batch['magno_image'].to(device)
            line_drawings = batch['line_drawing'].to(device)
            labels = batch['label'].to(device)

            # --- Latency Measurement ---
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            
            outputs = model(magno_images, line_drawings)
            
            end_time.record()
            torch.cuda.synchronize()
            total_latency += start_time.elapsed_time(end_time) # in milliseconds

            # --- Accuracy Calculation ---
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Calculate and Report Metrics ---
    num_samples = len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_latency = total_latency / num_samples

    # --- FLOPs Calculation ---
    # Get a single sample to analyze
    sample_magno = next(iter(val_loader))['magno_image'].to(device)
    sample_lines = next(iter(val_loader))['line_drawing'].to(device)
    # fvcore's FlopCountAnalysis needs inputs as a tuple
    inputs = (sample_magno, sample_lines) 
    flop_analyzer = FlopCountAnalysis(model, inputs)
    total_flops = flop_analyzer.total()

    print("\n--- Evaluation Results ---")
    print(f"Patch Percentage:       {args.patch_percentage * 100:.1f}%")
    print(f"Accuracy:               {accuracy * 100:.2f}%")
    print(f"Total GFLOPs:           {total_flops / 1e9:.2f} G")
    print(f"Avg. Latency per image: {avg_latency:.2f} ms")
    print("--------------------------")

    # --- 6. Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, val_dataset.class_names, args.results_dir, args.patch_percentage)
    
    # --- 7. Patch Selection Visualization (NEW) ---
    print("\n--- Generating Patch Selection Visualizations ---")
    visualize_patch_selection(
        model=model,
        val_loader=val_loader,
        device=device,
        output_dir=args.results_dir,
        num_samples=16,
        patch_percentage=args.patch_percentage
    )

    # --- 8. Patch Statistics Analysis (NEW) ---
    analyze_patch_selection_statistics(
        model=model,
        val_loader=val_loader,
        device=device,
        output_dir=args.results_dir,
        patch_percentage=args.patch_percentage
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the SelectiveMagnoViT model.")
    
    # Data and Model Paths
    parser.add_argument('--magno_dir', type=str, required=True, help="Path to the root of Magno images.")
    parser.add_argument('--lines_dir', type=str, required=True, help="Path to the root of Line Drawings.")
    parser.add_argument('--model_dir', type=str, default="models/checkpoints", help="Directory containing the trained model checkpoint.")
    parser.add_argument('--results_dir', type=str, default="results", help="Directory to save evaluation results like plots.")
    
    # Model Hyperparameters
    parser.add_argument('--patch_percentage', type=float, default=0.25, help="Patch percentage the model was trained with.")
    parser.add_argument('--img_size', type=int, default=256, help="Image size the model was trained on.")
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size")
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation (1 is best for latency).")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for data loading.")

    args = parser.parse_args()
    main(args)