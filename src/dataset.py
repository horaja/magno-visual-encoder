import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNetteDataset(Dataset):
    """
    Custom PyTorch Dataset for loading paired Magno images and Line Drawings.

    This dataset walks the directory structure to find corresponding pairs of
    preprocessed images, extracts class labels from the folder names, and
    applies the necessary transformations.
    """
    def __init__(self, magno_root, lines_root, split='train', transform=None):
        """
        Args:
            magno_root (str): Root directory for the Magno images (e.g., '.../data/preprocessed/magno_images').
            lines_root (str): Root directory for the Line Drawings (e.g., '.../data/preprocessed/line_drawings').
            split (str): The dataset split to load ('train' or 'val').
            transform (callable, optional): A torchvision transforms pipeline to be applied to the images.
        """
        self.magno_root = os.path.join(magno_root, split)
        self.lines_root = os.path.join(lines_root, split)
        self.split = split
        self.transform = transform

        if not os.path.isdir(self.magno_root):
            raise FileNotFoundError(f"Magno directory not found: {self.magno_root}")
        if not os.path.isdir(self.lines_root):
            raise FileNotFoundError(f"Line drawing directory not found: {self.lines_root}")

        # Find all line drawing images and create a list of their paths
        # We use this as the source of truth for pairing
        self.line_image_paths = sorted(glob(os.path.join(self.lines_root, '**', '*.png'), recursive=True))
        
        # Create a mapping from class name (e.g., 'n01440764') to an integer label
        self.class_names = sorted(os.listdir(self.lines_root))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}

        print(f"Found {len(self.line_image_paths)} images in '{self.split}' split.")
        print(f"Found {len(self.class_names)} classes.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.line_image_paths)

    def __getitem__(self, idx):
        """
        Fetches the a data sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the magno image, line drawing, and label.
        """
        # --- 1. Get Line Drawing Path and Load Image ---
        line_path = self.line_image_paths[idx]
        line_drawing = Image.open(line_path).convert('L') # Load as single-channel grayscale

        # --- 2. Construct the Corresponding Magno Image Path ---
        # Assumes filenames are consistent: ..._line.png -> ..._magno.png
        relative_path = os.path.relpath(line_path, self.lines_root)
        base_name = os.path.basename(line_path).replace('_line.png', '_magno.png')
        magno_path = os.path.join(self.magno_root, os.path.dirname(relative_path), base_name)
        
        magno_image = Image.open(magno_path).convert('RGB') # Load as 3-channel for ViT

        # --- 3. Extract the Label ---
        class_name = os.path.basename(os.path.dirname(line_path))
        label = self.class_to_idx[class_name]

        # --- 4. Apply Transformations ---
        if self.transform:
            magno_image = self.transform(magno_image)
            # Apply a simple ToTensor for the line drawing
            line_drawing = transforms.ToTensor()(line_drawing)
            line_drawing = 1.0 - line_drawing

        sample = {
            'magno_image': magno_image,
            'line_drawing': line_drawing,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return sample

# if __name__ == '__main__':
#     # Example usage and verification
#     MAGNO_DIR = 'data/preprocessed/magno_images'
#     LINES_DIR = 'data/preprocessed/line_drawings'

#     # Define transformations for the ViT input
#     vit_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Create a dataset instance for the 'train' split
#     train_dataset = ImageNetteDataset(
#         magno_root=MAGNO_DIR,
#         lines_root=LINES_DIR,
#         split='train',
#         transform=vit_transform
#     )

#     # Verify the dataset
#     if len(train_dataset) > 0:
#         print(f"\nSuccessfully created dataset with {len(train_dataset)} samples.")
#         sample = train_dataset[0]
#         magno_img_shape = sample['magno_image'].shape
#         line_img_shape = sample['line_drawing'].shape
#         label = sample['label']
        
#         print(f"Sample 0 shapes:")
#         print(f"  - Magno Image: {magno_img_shape}")
#         print(f"  - Line Drawing: {line_img_shape}")
#         print(f"  - Label: {label.item()} (Class: {train_dataset.idx_to_class[label.item()]})")

#         # Check for potential errors
#         assert magno_img_shape[0] == 3, "Magno image should be 3-channel"
#         assert line_img_shape[0] == 1, "Line drawing should be 1-channel"
#     else:
#         print("Dataset created, but it is empty. Check your data paths.")