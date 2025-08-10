import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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