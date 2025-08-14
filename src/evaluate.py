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
                
                # Get selected patch indices
                selected_indices = self.model.get_selected_patches_indices(line_drawings)
                
                for indices in selected_indices:
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
                selected_indices = self.model.get_selected_patches_indices(line_img)
                
                # Get prediction
                output = self.model(magno_img, line_img)
                _, pred = torch.max(output, 1)
                
                samples_data.append({
                    'magno': magno_img[0].cpu(),
                    'line': line_img[0].cpu(),
                    'patches': selected_indices[0].cpu(),
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


class AblationStudyEvaluator:
    """Conducts ablation study on patch selection strategies."""
    
    def __init__(self, model_path, device, val_loader, class_names, 
                 img_size=64, patch_size=4, num_classes=10):
        self.model_path = model_path
        self.device = device
        self.val_loader = val_loader
        self.class_names = class_names
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Load the base model weights once
        self.base_weights = torch.load(model_path, map_location=device)
        
        # Selection strategies to test
        self.strategies = {
            'random': 'Random Selection',
            'top_k': 'Top-K (Line-Guided)',
            'probabilistic': 'Probabilistic Sampling'
        }
        
        self.results = {}
    
    def create_model_with_strategy(self, strategy, patch_percentage):
        """Create a model with specific selection strategy and patch percentage."""
        model = SelectiveMagnoViT(
            patch_percentage=patch_percentage,
            num_classes=self.num_classes,
            img_size=self.img_size,
            patch_size=self.patch_size,
            selection_strategy=strategy
        ).to(self.device)
        
        # Load the pre-trained weights
        model.load_state_dict(self.base_weights, strict=False)
        model.eval()
        
        return model
    
    def evaluate_strategy_at_percentage(self, strategy, patch_percentage):
        """Evaluate a specific strategy at a given patch percentage."""
        model = self.create_model_with_strategy(strategy, patch_percentage)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, 
                             desc=f"{strategy} @ {patch_percentage:.1%}", 
                             leave=False):
                magno_images = batch['magno_image'].to(self.device)
                line_drawings = batch['line_drawing'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(magno_images, line_drawings)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        return accuracy
    
    def run_ablation_study(self, patch_percentages):
        """Run complete ablation study across strategies and percentages."""
        print("\n" + "="*60)
        print("ABLATION STUDY: Patch Selection Strategies")
        print("="*60)
        
        # Initialize results dictionary
        for strategy in self.strategies.keys():
            self.results[strategy] = {
                'percentages': patch_percentages,
                'accuracies': []
            }
        
        # Test each strategy at each percentage
        for strategy in self.strategies.keys():
            print(f"\nEvaluating {self.strategies[strategy]}...")
            
            for pct in patch_percentages:
                accuracy = self.evaluate_strategy_at_percentage(strategy, pct)
                self.results[strategy]['accuracies'].append(accuracy)
                print(f"  {pct:.1%} patches: {accuracy:.4f}")
        
        return self.results
    
    def plot_ablation_results(self, output_dir):
        """Create visualization of ablation study results."""
        plt.figure(figsize=(12, 8))
        
        # Define colors and styles for each strategy
        colors = {
            'random': '#FF6B6B',
            'top_k': '#4ECDC4',
            'probabilistic': '#45B7D1'
        }
        
        markers = {
            'random': 'o',
            'top_k': 's',
            'probabilistic': '^'
        }
        
        # Plot each strategy
        for strategy, label in self.strategies.items():
            percentages = [p * 100 for p in self.results[strategy]['percentages']]
            accuracies = [a * 100 for a in self.results[strategy]['accuracies']]
            
            plt.plot(percentages, accuracies, 
                    label=label,
                    color=colors[strategy],
                    marker=markers[strategy],
                    markersize=8,
                    linewidth=2.5,
                    alpha=0.9)
        
        # Styling
        plt.xlabel('Patch Percentage (%)', fontsize=14)
        plt.ylabel('Classification Accuracy (%)', fontsize=14)
        plt.title('Ablation Study: Patch Selection Strategies', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='lower right')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'ablation_study_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def save_results(self, output_dir):
        """Save ablation study results."""
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'strategies_tested': list(self.strategies.keys()),
            'results': self.results
        }
        
        output_path = os.path.join(output_dir, 'ablation_study_results.json')
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    def analyze_optimal_percentages(self, output_dir):
        """Analyze results to find optimal patch percentages based on different criteria."""
        
        analysis = {}
        
        # For each strategy
        for strategy in self.strategies.keys():
            percentages = self.results[strategy]['percentages']
            accuracies = self.results[strategy]['accuracies']
            
            # 1. Find the "knee point" - best trade-off between accuracy and efficiency
            knee_point = self._find_knee_point(percentages, accuracies)
            
            # 2. Find minimum percentage for target accuracy thresholds
            thresholds = self._find_accuracy_thresholds(percentages, accuracies)
            
            # 3. Calculate efficiency metrics
            efficiency = self._calculate_efficiency_metrics(percentages, accuracies)
            
            analysis[strategy] = {
                'knee_point': knee_point,
                'thresholds': thresholds,
                'efficiency': efficiency
            }
        
        # Create comprehensive analysis plot
        self._plot_analysis(analysis, output_dir)
        
        # Save analysis report
        self._save_analysis_report(analysis, output_dir)
        
        return analysis
    
    def _find_knee_point(self, percentages, accuracies):
        """Find the knee point using the elbow method."""
        from scipy.spatial.distance import euclidean
        
        # Normalize data
        x = np.array(percentages)
        y = np.array(accuracies)
        
        # Create line from first to last point
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        
        # Calculate distances from each point to the line
        distances = []
        for i in range(len(x)):
            p = np.array([x[i], y[i]])
            # Distance from point to line
            distance = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
            distances.append(distance)
        
        # Knee point is the one with maximum distance
        knee_idx = np.argmax(distances)
        
        return {
            'percentage': percentages[knee_idx],
            'accuracy': accuracies[knee_idx],
            'index': knee_idx
        }
    
    def _find_accuracy_thresholds(self, percentages, accuracies):
        """Find minimum patch % needed for various accuracy thresholds."""
        thresholds = {}
        target_accuracies = [0.9, 0.95, 0.98, 0.99]  # 90%, 95%, 98%, 99% of max accuracy
        
        max_accuracy = max(accuracies)
        
        for target in target_accuracies:
            target_acc = max_accuracy * target
            # Find first percentage that achieves this accuracy
            for i, acc in enumerate(accuracies):
                if acc >= target_acc:
                    thresholds[f'{int(target*100)}%_of_max'] = {
                        'percentage': percentages[i],
                        'accuracy': accuracies[i],
                        'target': target_acc
                    }
                    break
        
        return thresholds
    
    def _calculate_efficiency_metrics(self, percentages, accuracies):
        """Calculate efficiency metrics."""
        metrics = []
        
        for i, (pct, acc) in enumerate(zip(percentages, accuracies)):
            # Accuracy per patch percentage (efficiency score)
            efficiency_score = acc / pct if pct > 0 else 0
            
            # Relative efficiency compared to 100% patches
            relative_efficiency = acc / accuracies[-1] if accuracies[-1] > 0 else 0
            
            # Speedup factor (inverse of patch percentage)
            theoretical_speedup = 1.0 / pct if pct > 0 else 0
            
            metrics.append({
                'percentage': pct,
                'accuracy': acc,
                'efficiency_score': efficiency_score,
                'relative_accuracy': relative_efficiency,
                'theoretical_speedup': theoretical_speedup,
                'efficiency_ratio': relative_efficiency * theoretical_speedup
            })
        
        return metrics
    
    def _plot_analysis(self, analysis, output_dir):
        """Create comprehensive analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Accuracy curves with knee points
        ax = axes[0, 0]
        for strategy in self.strategies.keys():
            percentages = [p * 100 for p in self.results[strategy]['percentages']]
            accuracies = [a * 100 for a in self.results[strategy]['accuracies']]
            knee = analysis[strategy]['knee_point']
            
            ax.plot(percentages, accuracies, label=self.strategies[strategy], 
                   marker='o', linewidth=2)
            # Mark knee point
            ax.scatter([knee['percentage'] * 100], [knee['accuracy'] * 100], 
                      s=200, marker='*', edgecolors='red', linewidths=2, 
                      facecolors='none', zorder=5)
        
        ax.set_xlabel('Patch Percentage (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Patch % (★ = Knee Points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Efficiency scores
        ax = axes[0, 1]
        for strategy in self.strategies.keys():
            efficiency = analysis[strategy]['efficiency']
            percentages = [e['percentage'] * 100 for e in efficiency]
            scores = [e['efficiency_score'] * 100 for e in efficiency]
            
            ax.plot(percentages, scores, label=self.strategies[strategy], 
                   marker='s', linewidth=2)
        
        ax.set_xlabel('Patch Percentage (%)')
        ax.set_ylabel('Efficiency Score (Acc/Patch%)')
        ax.set_title('Efficiency Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy vs Speedup trade-off
        ax = axes[0, 2]
        for strategy in self.strategies.keys():
            efficiency = analysis[strategy]['efficiency']
            speedups = [e['theoretical_speedup'] for e in efficiency]
            accuracies = [e['accuracy'] * 100 for e in efficiency]
            
            ax.plot(speedups, accuracies, label=self.strategies[strategy], 
                   marker='^', linewidth=2)
        
        ax.set_xlabel('Theoretical Speedup')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Speedup Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 4: Marginal accuracy gain
        ax = axes[1, 0]
        for strategy in self.strategies.keys():
            percentages = self.results[strategy]['percentages']
            accuracies = self.results[strategy]['accuracies']
            
            # Calculate marginal gains
            marginal_gains = []
            for i in range(1, len(accuracies)):
                gain = (accuracies[i] - accuracies[i-1]) / (percentages[i] - percentages[i-1])
                marginal_gains.append(gain)
            
            ax.plot([p * 100 for p in percentages[1:]], 
                   [g * 100 for g in marginal_gains],
                   label=self.strategies[strategy], marker='o', linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Patch Percentage (%)')
        ax.set_ylabel('Marginal Accuracy Gain (%/patch%)')
        ax.set_title('Diminishing Returns Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Relative performance
        ax = axes[1, 1]
        baseline = self.results['top_k']['accuracies']
        
        for strategy in self.strategies.keys():
            percentages = [p * 100 for p in self.results[strategy]['percentages']]
            relative = [(a/b) * 100 for a, b in 
                       zip(self.results[strategy]['accuracies'], baseline)]
            
            ax.plot(percentages, relative, label=self.strategies[strategy], 
                   marker='d', linewidth=2)
        
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Patch Percentage (%)')
        ax.set_ylabel('Relative to Top-K (%)')
        ax.set_title('Relative Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Recommendation zones
        ax = axes[1, 2]
        best_strategy = 'top_k'  # Usually performs best
        percentages = [p * 100 for p in self.results[best_strategy]['percentages']]
        accuracies = [a * 100 for a in self.results[best_strategy]['accuracies']]
        
        ax.plot(percentages, accuracies, 'b-', linewidth=3, label='Performance')
        
        # Add recommendation zones
        ax.axvspan(10, 30, alpha=0.2, color='red', label='Low Quality')
        ax.axvspan(30, 50, alpha=0.2, color='yellow', label='Aggressive')
        ax.axvspan(50, 70, alpha=0.2, color='lightgreen', label='Balanced')
        ax.axvspan(70, 100, alpha=0.2, color='blue', label='Conservative')
        
        # Mark key points
        knee = analysis[best_strategy]['knee_point']
        ax.scatter([knee['percentage'] * 100], [knee['accuracy'] * 100], 
                  s=200, marker='*', color='red', label='Recommended', zorder=5)
        
        ax.set_xlabel('Patch Percentage (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Deployment Recommendations')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Patch Percentage Selection Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'patch_selection_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _save_analysis_report(self, analysis, output_dir):
        """Save detailed analysis report with recommendations."""
        report_path = os.path.join(output_dir, 'optimal_percentage_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OPTIMAL PATCH PERCENTAGE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall recommendation
            best_strategy = 'top_k'  # Usually the best
            knee = analysis[best_strategy]['knee_point']
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Recommended Patch Percentage: {knee['percentage']:.1%}\n")
            f.write(f"Expected Accuracy: {knee['accuracy']:.4f}\n")
            f.write(f"Strategy: {self.strategies[best_strategy]}\n\n")
            
            # Detailed analysis for each strategy
            for strategy in self.strategies.keys():
                f.write(f"\n{self.strategies[strategy].upper()}\n")
                f.write("-"*40 + "\n")
                
                knee = analysis[strategy]['knee_point']
                f.write(f"Knee Point (Optimal Trade-off): {knee['percentage']:.1%} "
                       f"(Accuracy: {knee['accuracy']:.4f})\n")
                
                f.write("\nAccuracy Thresholds:\n")
                for threshold_name, threshold_data in analysis[strategy]['thresholds'].items():
                    f.write(f"  {threshold_name}: {threshold_data['percentage']:.1%} patches "
                           f"(Acc: {threshold_data['accuracy']:.4f})\n")
                
                # Find most efficient point
                efficiency = analysis[strategy]['efficiency']
                best_efficiency = max(efficiency, key=lambda x: x['efficiency_ratio'])
                f.write(f"\nMost Efficient Point: {best_efficiency['percentage']:.1%} "
                       f"(Efficiency Ratio: {best_efficiency['efficiency_ratio']:.2f})\n")
            
            # Recommendations for different use cases
            f.write("\n" + "="*80 + "\n")
            f.write("USE CASE RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            recommendations = {
                'Real-time (Latency Critical)': 0.3,
                'Balanced (Mobile/Edge)': knee['percentage'],
                'High Accuracy (Server)': 0.7,
                'Maximum Accuracy': 1.0
            }
            
            for use_case, recommended_pct in recommendations.items():
                # Find closest actual percentage
                percentages = self.results[best_strategy]['percentages']
                closest_idx = np.argmin(np.abs(np.array(percentages) - recommended_pct))
                actual_pct = percentages[closest_idx]
                actual_acc = self.results[best_strategy]['accuracies'][closest_idx]
                
                f.write(f"{use_case:30} -> {actual_pct:>6.1%} patches "
                       f"(Accuracy: {actual_acc:.4f}, "
                       f"Speedup: {1/actual_pct:.1f}x)\n")
        
        print(f"\nDetailed analysis report saved to: {report_path}")


def standard_evaluation(args):
    """Standard evaluation mode (backward compatible with existing scripts)."""
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
    
    # Try to load model - handle both naming conventions
    if args.model_path:
        model_path = args.model_path
    else:
        # Try to find the model with different naming patterns
        possible_paths = [
            os.path.join(args.model_dir, f"best_model_pp{args.patch_percentage}.pth"),
            os.path.join(args.model_dir, f"best_model_pp1.0.pth"),  # If trained at 100%
            os.path.join(args.model_dir, "best_model_pp1.pth")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Could not find model in {args.model_dir}")
    
    # Load the model weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # Set the desired patch percentage for evaluation
    model.set_patch_percentage(args.patch_percentage)
    
    print(f"Model loaded from {model_path}")
    print(f"Evaluating at {args.patch_percentage:.1%} patch percentage")
    
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


def ablation_study(args):
    """Run ablation study mode."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.plots_dir if args.plots_dir else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
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
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Get model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(args.model_dir, "best_model_pp1.0.pth")
    
    # Define patch percentages to test
    patch_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Initialize evaluator
    evaluator = AblationStudyEvaluator(
        model_path=model_path,
        device=device,
        val_loader=val_loader,
        class_names=val_dataset.class_names,
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=len(val_dataset.class_names)
    )
    
    # Run ablation study
    results = evaluator.run_ablation_study(patch_percentages)
    
    # Analyze optimal percentages
    evaluator.analyze_optimal_percentages(output_dir)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_ablation_results(output_dir)
    
    # Save results
    evaluator.save_results(output_dir)


def main(args):
    """Main evaluation function."""
    if args.ablation_study:
        ablation_study(args)
    else:
        standard_evaluation(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SelectiveMagnoViT model")
    
    # Data paths
    parser.add_argument('--magno_dir', type=str, required=True)
    parser.add_argument('--lines_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None,
                       help='Direct path to model checkpoint (overrides model_dir)')
    
    # Output paths
    parser.add_argument('--plots_dir', type=str, required=True)
    parser.add_argument('--results_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (used for ablation study)')
    
    # Model parameters
    parser.add_argument('--patch_percentage', type=float, default=0.3)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    
    # Evaluation settings
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--visualize', type=str, default='false')
    parser.add_argument('--ablation_study', action='store_true',
                       help='Run ablation study instead of standard evaluation')
    
    args = parser.parse_args()
    args.visualize = args.visualize.lower() == 'true'
    
    main(args)