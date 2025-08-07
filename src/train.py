import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time

# Import our custom modules
from model import SelectiveMagnoViT
from dataset import ImageNetteDataset

# Import a standard optimizer and learning rate scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def main(args):
    """
    Main function to orchestrate the training and validation process.
    """
    # --- 1. Device and Directory Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the output directory for saving models if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Data Loading ---
    # Define transformations for the ViT input (Magno images)
    # These should match the standard transformations for ViT models
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)), # Randomly zoom and crop
        transforms.RandomHorizontalFlip(), # Flip images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly change brightness/contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), # Keep validation consistent
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets for training and validation
    print("Loading training data...")
    train_dataset = ImageNetteDataset(
        magno_root=args.magno_dir,
        lines_root=args.lines_dir,
        split='train',
        transform=train_transform
    )
    
    print("Loading validation data...")
    val_dataset = ImageNetteDataset(
        magno_root=args.magno_dir,
        lines_root=args.lines_dir,
        split='val',
        transform=val_transform
    )

    # Create DataLoaders
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
    
    num_classes = len(train_dataset.class_names)
    print(f"Number of classes: {num_classes}")

    # --- 3. Model, Optimizer, and Loss Function ---
    print("Initializing model...")
    model = SelectiveMagnoViT(
        patch_percentage=args.patch_percentage,
        num_classes=num_classes,
        img_size=args.img_size
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.03)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # Reduce LR every 10 epochs
    criterion = nn.CrossEntropyLoss() # Standard loss for classification

    # --- 4. Training and Validation Loop ---
    best_val_accuracy = 0.0
    print("--- Starting Training ---")

    for epoch in range(args.epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            magno_images = batch['magno_image'].to(device)
            line_drawings = batch['line_drawing'].to(device)
            labels = batch['label'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(magno_images, line_drawings)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * magno_images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / total_samples
        train_accuracy = correct_predictions / total_samples

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in progress_bar_val:
                magno_images = batch['magno_image'].to(device)
                line_drawings = batch['line_drawing'].to(device)
                labels = batch['label'].to(device)

                outputs = model(magno_images, line_drawings)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * magno_images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        val_loss = val_loss / total_samples
        val_accuracy = correct_predictions / total_samples
        
        scheduler.step() # Update learning rate

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | "
              f"Duration: {epoch_duration:.2f}s")

        # --- 5. Checkpointing ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(args.output_dir, f"best_model_pp{args.patch_percentage}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path} with accuracy: {val_accuracy:.4f}")

    print("--- Finished Training ---")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the SelectiveMagnoViT model.")
    
    # Data paths
    parser.add_argument('--magno_dir', type=str, required=True, help="Path to the root directory of Magno images.")
    parser.add_argument('--lines_dir', type=str, required=True, help="Path to the root directory of Line Drawings.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save model checkpoints.")
    
    # Model hyperparameters
    parser.add_argument('--patch_percentage', type=float, default=0.25, help="Percentage of patches to select (e.g., 0.25 for 25%).")
    parser.add_argument('--img_size', type=int, default=256, help="Image size to train the model on.")
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for data loading.")

    args = parser.parse_args()
    main(args)