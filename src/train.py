import os
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import time
import random

from model import SelectiveMagnoViT
from dataset import ImageNetteDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    """Handles training and validation of SelectiveMagnoViT model."""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup tensorboard
        if config.get('tensorboard_dir'):
            self.writer = SummaryWriter(config['tensorboard_dir'])
        else:
            self.writer = None
        
        # Training state
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.epoch = 0
        
        # Dynamic patch percentage settings
        self.use_dynamic_patches = config.get('use_dynamic_patches', False)
        self.dynamic_patch_range = config.get('dynamic_patch_range', [0.3, 1.0])
        self.dynamic_patch_schedule = config.get('dynamic_patch_schedule', 'random')  # 'random' or 'curriculum'
        
    def get_patch_percentage_for_epoch(self):
        """Get patch percentage for current epoch based on schedule."""
        if not self.use_dynamic_patches:
            return self.model.patch_percentage
        
        if self.dynamic_patch_schedule == 'curriculum':
            # Start with 100% and gradually decrease
            progress = self.epoch / self.config['epochs']
            min_pct, max_pct = self.dynamic_patch_range
            return max_pct - (max_pct - min_pct) * progress
        
        elif self.dynamic_patch_schedule == 'random':
            # Random sampling between min and max
            min_pct, max_pct = self.dynamic_patch_range
            return random.uniform(min_pct, max_pct)
        
        else:
            return self.model.patch_percentage
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Get patch percentage for this epoch
        epoch_patch_percentage = self.get_patch_percentage_for_epoch()
        self.model.set_patch_percentage(epoch_patch_percentage)
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch+1} [Train] PP={epoch_patch_percentage:.1%}"
        )
        
        for batch in progress_bar:
            magno_images = batch['magno_image'].to(self.device)
            line_drawings = batch['line_drawing'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(magno_images, line_drawings)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * magno_images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss = running_loss / total
        train_accuracy = correct / total
        
        return train_loss, train_accuracy, epoch_patch_percentage
    
    def validate(self, patch_percentage=None):
        """
        Validate the model.
        
        Args:
            patch_percentage (float, optional): Override patch percentage for validation.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use specified percentage or model's current percentage
        if patch_percentage is not None:
            original_percentage = self.model.patch_percentage
            self.model.set_patch_percentage(patch_percentage)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")
            for batch in progress_bar:
                magno_images = batch['magno_image'].to(self.device)
                line_drawings = batch['line_drawing'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(magno_images, line_drawings)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * magno_images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Restore original percentage if changed
        if patch_percentage is not None:
            self.model.set_patch_percentage(original_percentage)
        
        val_loss = running_loss / total
        val_accuracy = correct / total
        
        return val_loss, val_accuracy
    
    def validate_multiple_percentages(self):
        """Validate at multiple patch percentages."""
        test_percentages = [0.3, 0.5, 0.7, 1.0]
        results = {}
        
        for pct in test_percentages:
            val_loss, val_acc = self.validate(patch_percentage=pct)
            results[pct] = {'loss': val_loss, 'accuracy': val_acc}
            print(f"    Val @ {pct:.0%} patches: Acc={val_acc:.4f}, Loss={val_loss:.4f}")
        
        return results
    
    def save_checkpoint(self, path, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'model_info': self.model.get_model_info()
        }
        torch.save(checkpoint, path)
        
        if is_best:
            # Save just the model state dict for easy loading
            best_path = path.replace('.pth', '_best.pth')
            torch.save(self.model.state_dict(), best_path)
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("="*60)
        print(f"Model Info: {self.model.get_model_info()}")
        print("="*60 + "\n")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss, train_acc, train_patch_pct = self.train_epoch()
            
            # Validate at training patch percentage
            val_loss, val_acc = self.validate()
            
            # Optionally validate at multiple percentages
            if epoch % 10 == 0 and self.config.get('validate_multiple_percentages', False):
                print("  Validating at multiple patch percentages:")
                multi_val_results = self.validate_multiple_percentages()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_rate', 
                                      self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Patch_percentage/train', train_patch_pct, epoch)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config['epochs']} "
                  f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} @ {train_patch_pct:.1%} patches "
                  f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} "
                  f"| Time: {epoch_time:.2f}s")
            
            # Check if best model
            is_best = val_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                print(f"  → New best model! Accuracy: {val_acc:.4f}")
                
                # Save best model
                best_path = os.path.join(
                    self.config['output_dir'],
                    f"best_model_pp{self.config['patch_percentage']}.pth"
                )
                torch.save(self.model.state_dict(), best_path)
            else:
                self.patience_counter += 1
            
            # # Save periodic checkpoint
            # if epoch % 10 == 0:
            #     checkpoint_path = os.path.join(
            #         self.config['output_dir'],
            #         f"checkpoint_epoch{epoch}_pp{self.config['patch_percentage']}.pth"
            #     )
            #     self.save_checkpoint(checkpoint_path, is_best)
                
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        print("="*60)
        
        if self.writer:
            self.writer.close()
        
        return self.best_val_accuracy


def main(args):
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # Setup data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ImageNetteDataset(
        magno_root=args.magno_dir,
        lines_root=args.lines_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = ImageNetteDataset(
        magno_root=args.magno_dir,
        lines_root=args.lines_dir,
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    num_classes = len(train_dataset.class_names)
    print(f"Number of classes: {num_classes}")
    
    model = SelectiveMagnoViT(
        patch_percentage=args.patch_percentage,
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        vit_model_name=args.vit_model_name
    ).to(device)
    
    # Setup configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'patch_percentage': args.patch_percentage,
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'vit_model_name': args.vit_model_name,
        'num_classes': num_classes,
        'output_dir': args.output_dir,
        'tensorboard_dir': args.tensorboard_dir,
        'use_dynamic_patches': args.use_dynamic_patches,
        'dynamic_patch_range': args.dynamic_patch_range,
        'dynamic_patch_schedule': args.dynamic_patch_schedule,
        'validate_multiple_percentages': args.validate_multiple_percentages,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save configuration
    config_path = os.path.join(args.output_dir, f"config_pp{args.patch_percentage}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Initialize trainer and start training
    trainer = Trainer(model, train_loader, val_loader, config, device)
    best_accuracy = trainer.train()
    
    return best_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SelectiveMagnoViT model")
    
    # Data paths
    parser.add_argument('--magno_dir', type=str, required=True)
    parser.add_argument('--lines_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    
    # Model parameters
    parser.add_argument('--patch_percentage', type=float, default=0.3)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--vit_model_name', type=str, 
                       default='vit_tiny_patch16_224.augreg_in21k')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Dynamic patch percentage settings
    parser.add_argument('--use_dynamic_patches', action='store_true',
                       help='Use dynamic patch percentages during training')
    parser.add_argument('--dynamic_patch_range', type=float, nargs=2, default=[0.3, 1.0],
                       help='Min and max patch percentages for dynamic training')
    parser.add_argument('--dynamic_patch_schedule', type=str, default='random',
                       choices=['random', 'curriculum'],
                       help='Schedule for dynamic patch percentage')
    parser.add_argument('--validate_multiple_percentages', action='store_true',
                       help='Validate at multiple patch percentages during training')
    
    args = parser.parse_args()
    main(args)