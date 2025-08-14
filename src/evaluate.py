import os
import argparse
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from fvcore.nn import FlopCountAnalysis

from model import SelectiveMagnoViT
from dataset import ImageNetteDataset
from torchvision import transforms


class ModelEvaluator:
    """Handles all evaluation tasks for SelectiveMagnoViT model."""
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.results = {}
        
    def evaluate_performance(self, val_loader):
        """Evaluate model accuracy, latency, and FLOPs."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_latency = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                magno_images = batch['magno_image'].to(self.device)
                line_drawings = batch['line_drawing'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Measure latency
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    
                outputs = self.model(magno_images, line_drawings)
                
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    total_latency += start_event.elapsed_time(end_event)
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_latency = total_latency / len(all_labels) if torch.cuda.is_available() else 0
        
        # Calculate FLOPs
        sample_batch = next(iter(val_loader))
        sample_magno = sample_batch['magno_image'][:1].to(self.device)
        sample_lines = sample_batch['line_drawing'][:1].to(self.device)
        flops = FlopCountAnalysis(self.model, (sample_magno, sample_lines)).total()
        
        self.results['accuracy'] = accuracy
        self.results['avg_latency_ms'] = avg_latency
        self.results['total_gflops'] = flops / 1e9
        self.results['predictions'] = all_preds
        self.results['labels'] = all_labels
        
        return accuracy, avg_latency, flops / 1e9
    
    def generate_confusion_matrix(self, output_path):
        """Generate and save confusion matrix."""
        cm = confusion_matrix(self.results['labels'], self.results['predictions'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Accuracy: {self.results["accuracy"]*100:.2f}%)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return cm
    
    def analyze_patch_selection(self, val_loader, num_samples=16):
        """Analyze patch selection patterns."""
        self.model.eval()
        
        img_size = next(iter(val_loader))['magno_image'].shape[-1]
        num_patches = (img_size // self.model.patch_size) ** 2
        patch_selection_counts = torch.zeros(num_patches)
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Analyzing patches"):
                line_drawings = batch['line_drawing'].to(self.device)
                patch_scores = self.model.scorer(line_drawings)
                
                k = int(patch_scores.shape[1] * self.model.selector.patch_percentage)
                _, top_k_indices = torch.topk(patch_scores, k=k, dim=1)
                
                for indices in top_k_indices:
                    for idx in indices:
                        patch_selection_counts[idx] += 1
                
                total_samples += line_drawings.shape[0]
        
        patch_frequencies = patch_selection_counts / total_samples
        
        self.results['patch_frequencies'] = patch_frequencies.numpy()
        self.results['patch_stats'] = {
            'mean_frequency': float(patch_frequencies.mean()),
            'std_frequency': float(patch_frequencies.std()),
            'max_frequency': float(patch_frequencies.max()),
            'min_frequency': float(patch_frequencies.min()),
            'most_selected_patch': int(patch_frequencies.argmax()),
            'least_selected_patch': int(patch_frequencies.argmin())
        }
        
        return patch_frequencies

    def visualize_patch_selection(self, val_loader, output_dir, num_samples=16):
        """Create comprehensive patch selection visualizations."""
        self.model.eval()
        
        samples_data = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_samples:
                    break
                    
                magno_img = batch['magno_image'].to(self.device)
                line_img = batch['line_drawing'].to(self.device)
                label = batch['label'].to(self.device)
                
                # Get patch information
                patch_scores = self.model.scorer(line_img)
                k = int(patch_scores.shape[1] * self.model.selector.patch_percentage)
                _, top_k_indices = torch.topk(patch_scores, k=k, dim=1)
                
                # Get prediction
                output = self.model(magno_img, line_img)
                _, pred = torch.max(output, 1)
                
                samples_data.append({
                    'magno': magno_img[0].cpu(),
                    'line': line_img[0].cpu(),
                    'patches': top_k_indices[0].cpu(),
                    'scores': patch_scores[0].cpu(),
                    'pred': pred[0].cpu().item(),
                    'label': label[0].cpu().item()
                })
        
        # Create visualization
        self._create_patch_visualization(samples_data, output_dir)
        
    def _create_patch_visualization(self, samples_data, output_dir):
        """Helper to create patch visualization figure."""
        num_samples = len(samples_data)
        fig = plt.figure(figsize=(20, 4 * num_samples))
        gs = GridSpec(num_samples, 5, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, data in enumerate(samples_data):
            # Original image
            ax1 = fig.add_subplot(gs[idx, 0])
            img = self._denormalize_image(data['magno'])
            ax1.imshow(img)
            ax1.set_title(f'Original\nTrue: {self.class_names[data["label"]]}', fontsize=10)
            ax1.axis('off')
            
            # Line drawing
            ax2 = fig.add_subplot(gs[idx, 1])
            ax2.imshow(data['line'].squeeze(), cmap='gray')
            ax2.set_title(f'Line Drawing\nPred: {self.class_names[data["pred"]]}', fontsize=10)
            ax2.axis('off')
            
            # Selected patches
            ax3 = fig.add_subplot(gs[idx, 2])
            masked = self._create_masked_image(data['magno'], data['patches'], self.model.patch_size)
            ax3.imshow(masked)
            ax3.set_title(f'Selected Patches', fontsize=10)
            ax3.axis('off')
            
            # Patch overlay
            ax4 = fig.add_subplot(gs[idx, 3])
            overlay = self._create_patch_overlay(data['magno'], data['patches'], self.model.patch_size)
            ax4.imshow(overlay)
            ax4.set_title('Patch Grid', fontsize=10)
            ax4.axis('off')
            
            # Importance heatmap
            ax5 = fig.add_subplot(gs[idx, 4])
            heatmap = self._create_importance_heatmap(
                data['scores'], data['patches'], 
                self.model.patch_size, data['magno'].shape[-1]
            )
            ax5.imshow(heatmap, cmap='hot')
            ax5.set_title('Importance Scores', fontsize=10)
            ax5.axis('off')
        
        output_path = os.path.join(output_dir, 'patch_selection_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _denormalize_image(self, tensor):
        """Denormalize image tensor."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = tensor * std + mean
        img = torch.clamp(img, 0, 1)
        return img.permute(1, 2, 0).numpy()
    
    def _create_masked_image(self, image_tensor, selected_indices, patch_size):
        """Create masked image showing selected patches."""
        img = self._denormalize_image(image_tensor.clone())
        H, W = img.shape[:2]
        patches_per_row = W // patch_size
        
        mask = np.zeros((H, W))
        for idx in selected_indices:
            row = idx // patches_per_row
            col = idx % patches_per_row
            y_start = row * patch_size
            x_start = col * patch_size
            mask[y_start:y_start + patch_size, x_start:x_start + patch_size] = 1
        
        return img * mask[:, :, np.newaxis]
    
    def _create_patch_overlay(self, image_tensor, selected_indices, patch_size):
        """Create overlay with patch grid."""
        img = self._denormalize_image(image_tensor.clone())
        H, W = img.shape[:2]
        patches_per_row = W // patch_size
        
        # Create copy for overlay
        overlay = img.copy()
        
        # Draw grid
        for i in range(0, H, patch_size):
            overlay[i:i+1, :] = overlay[i:i+1, :] * 0.7
        for j in range(0, W, patch_size):
            overlay[:, j:j+1] = overlay[:, j:j+1] * 0.7
        
        # Highlight selected patches
        for idx in selected_indices:
            row = idx // patches_per_row
            col = idx % patches_per_row
            y_start = row * patch_size
            x_start = col * patch_size
            # Draw green border
            overlay[y_start:y_start+2, x_start:x_start+patch_size, 1] = 1
            overlay[y_start+patch_size-2:y_start+patch_size, x_start:x_start+patch_size, 1] = 1
            overlay[y_start:y_start+patch_size, x_start:x_start+2, 1] = 1
            overlay[y_start:y_start+patch_size, x_start+patch_size-2:x_start+patch_size, 1] = 1
        
        return overlay
    
    def _create_importance_heatmap(self, scores, selected_indices, patch_size, image_size):
        """Create importance score heatmap."""
        num_patches_per_dim = image_size // patch_size
        heatmap = scores.reshape(num_patches_per_dim, num_patches_per_dim).numpy()
        
        # Upscale for visualization
        heatmap_large = np.repeat(np.repeat(heatmap, patch_size, axis=0), patch_size, axis=1)
        
        return heatmap_large
    
    def save_results(self, results_file, config):
        """Save all results to a JSON file."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'performance': {
                'accuracy': float(self.results['accuracy']),
                'avg_latency_ms': float(self.results['avg_latency_ms']),
                'total_gflops': float(self.results['total_gflops'])
            },
            'patch_statistics': self.results.get('patch_stats', {}),
            'classification_report': classification_report(
                self.results['labels'], 
                self.results['predictions'],
                target_names=self.class_names,
                output_dict=True
            )
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Also create a human-readable summary
        summary_file = results_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SELECTIVEMAGNOVIT EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-"*30 + "\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nPERFORMANCE METRICS:\n")
            f.write("-"*30 + "\n")
            f.write(f"  Accuracy: {self.results['accuracy']*100:.2f}%\n")
            f.write(f"  Avg Latency: {self.results['avg_latency_ms']:.2f} ms\n")
            f.write(f"  Total GFLOPs: {self.results['total_gflops']:.2f}\n")
            
            if 'patch_stats' in self.results:
                f.write("\nPATCH SELECTION STATISTICS:\n")
                f.write("-"*30 + "\n")
                for key, value in self.results['patch_stats'].items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")


def main(args):
    """Main evaluation function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = ImageNetteDataset(
        magno_root=args.magno_dir,
        lines_root=args.lines_dir,
        split='val',
        transform=transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use 1 for accurate latency measurement
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load model
    num_classes = len(val_dataset.class_names)
    model = SelectiveMagnoViT(
        patch_percentage=args.patch_percentage,
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size
    ).to(device)
    
    model_path = os.path.join(args.model_dir, f"best_model_pp{args.patch_percentage}.pth")
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device, val_dataset.class_names)
    
    # Run evaluation
    print("\nEvaluating performance...")
    accuracy, latency, gflops = evaluator.evaluate_performance(val_loader)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Avg Latency: {latency:.2f} ms")
    print(f"  GFLOPs: {gflops:.2f}")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(args.plots_dir, f"confusion_matrix_pp{args.patch_percentage}.png")
    evaluator.generate_confusion_matrix(cm_path)
    
    # Analyze patch selection
    print("\nAnalyzing patch selection...")
    evaluator.analyze_patch_selection(val_loader)
    
    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        evaluator.visualize_patch_selection(val_loader, args.plots_dir)
        
        # Plot patch frequency heatmap
        patch_freq = evaluator.results['patch_frequencies']
        patches_per_dim = int(np.sqrt(len(patch_freq)))
        freq_grid = patch_freq.reshape(patches_per_dim, patches_per_dim)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(freq_grid, cmap='viridis')
        plt.colorbar(label='Selection Frequency')
        plt.title('Patch Selection Frequency Heatmap')
        plt.xlabel('Patch Column')
        plt.ylabel('Patch Row')
        freq_path = os.path.join(args.plots_dir, f"patch_frequency_pp{args.patch_percentage}.png")
        plt.savefig(freq_path, dpi=150)
        plt.close()
    
    # Save results
    config = {
        'model_path': model_path,
        'patch_percentage': args.patch_percentage,
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'num_classes': num_classes
    }
    
    results_file = args.results_file.replace('.txt', '.json')
    evaluator.save_results(results_file, config)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SelectiveMagnoViT model")
    
    # Data paths
    parser.add_argument('--magno_dir', type=str, required=True)
    parser.add_argument('--lines_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    
    # Output paths
    parser.add_argument('--plots_dir', type=str, required=True)
    parser.add_argument('--results_file', type=str, required=True)
    
    # Model parameters
    parser.add_argument('--patch_percentage', type=float, default=0.3)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    
    # Evaluation settings
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--visualize', type=str, default='false')
    
    args = parser.parse_args()
    args.visualize = args.visualize.lower() == 'true'
    
    main(args)