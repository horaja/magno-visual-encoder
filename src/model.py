import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PatchImportanceScorer(nn.Module):
    """
    Module 2: Calculates a "density score" for each patch location.
    
    This module uses average pooling to efficiently sum the pixel intensities
    of a line drawing within each potential patch region.
    """
    def __init__(self, patch_size):
        """
        Args:
            patch_size (int): The size of the patches (e.g., 16 for a ViT-Base/16).
        """
        super().__init__()
        self.patch_size = patch_size
        # Use average pooling as a highly efficient way to sum pixel values in a patch
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, line_drawing):
        """
        Args:
            line_drawing (torch.Tensor): A batch of single-channel line drawings
                                         of shape (B, 1, H, W).
        
        Returns:
            torch.Tensor: A tensor of shape (B, num_patches) containing the
                          density score for each patch.
        """
        # The line drawing should have values [0, 1].
        # The average pixel value * patch_area is the sum of pixel values.
        # (B, 1, H, W) -> (B, 1, H/P, W/P)
        avg_pool_scores = self.pool(line_drawing)
        
        # We can just use the average value as a score, as the area is constant.
        # (B, 1, H/P, W/P) -> (B, num_patches)
        scores = avg_pool_scores.flatten(start_dim=1)
        
        return scores

class TopKPatchSelector(nn.Module):
    """
    Module 3 & 4: Selects top-K patches from the Magno image based on scores
                  and combines them with their original positional embeddings.
    """
    def __init__(self, patch_percentage):
        """
        Args:
            patch_percentage (float): The fraction of patches to select (e.g., 0.25 for 25%).
        """
        super().__init__()
        if not (0 < patch_percentage <= 1.0):
            raise ValueError("patch_percentage must be between 0 and 1.")
        self.patch_percentage = patch_percentage

    def forward(self, magno_patches, vit_positional_embedding, scores):
        """
        Args:
            magno_patches (torch.Tensor): All patches from the Magno image,
                                          shape (B, num_patches, patch_dim).
            vit_positional_embedding (torch.Tensor): The ViT's standard positional
                                                     embeddings, shape (1, num_patches + 1, embed_dim).
            scores (torch.Tensor): The density scores from the PatchImportanceScorer,
                                   shape (B, num_patches).
        
        Returns:
            torch.Tensor: A sparse sequence of selected patch embeddings with their
                          positional encodings added, shape (B, num_selected_patches, embed_dim).
        """
        B, N, D = magno_patches.shape
        k = int(N * self.patch_percentage)

        # Get the indices of the top-k scores for each image in the batch
        # topk returns (values, indices)
        _, top_k_indices = torch.topk(scores, k=k, dim=1) # (B, k)

        # Gather the top-k patches using the indices
        # We need to expand indices to match the dimension of the patches
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_patches = torch.gather(magno_patches, 1, top_k_indices_expanded)

        # --- Positional Embedding ---
        # The ViT positional embedding includes a token for the [CLS] token,
        # so we skip it by indexing from [:, 1:, :]
        pos_embed_patches = vit_positional_embedding[:, 1:, :] # (1, num_patches, embed_dim)
        
        # Gather the positional embeddings corresponding to the selected patches
        # We need to expand indices for this gathering operation as well
        pos_embed_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, pos_embed_patches.size(-1))
        selected_pos_embed = torch.gather(pos_embed_patches.expand(B, -1, -1), 1, pos_embed_indices)

        # Add the positional embeddings to the selected patches
        return selected_patches + selected_pos_embed


class SelectiveMagnoViT(nn.Module):
    """
    SelectiveMagnoViT: A Vision Transformer that selectively processes patches
    based on importance scores from line drawings.
    
    Args:
        patch_percentage (float): Fraction of patches to select (0-1).
        num_classes (int): Number of output classes.
        img_size (int): Input image size.
        patch_size (int): Size of each patch.
        vit_model_name (str): Name of the pre-trained ViT model from timm.
        embed_dim (int, optional): Embedding dimension for custom ViT.
    """
    
    def __init__(self, 
                 patch_percentage=0.25,
                 num_classes=10,
                 img_size=64,
                 patch_size=4,
                 vit_model_name='vit_tiny_patch16_224.augreg_in21k',
                 embed_dim=None):
        super().__init__()
        
        # Validate inputs
        if not 0 < patch_percentage <= 1.0:
            raise ValueError(f"patch_percentage must be in (0, 1], got {patch_percentage}")
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        
        # Store configuration
        self.patch_percentage = patch_percentage
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Load ViT backbone
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        
        # Get embedding dimension
        if embed_dim is None:
            embed_dim = self.vit.embed_dim
        
        # Replace patch embedding layer for custom image size
        self.vit.patch_embed = timm.models.vision_transformer.PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )
        
        # Update positional embeddings
        num_patches = self.vit.patch_embed.num_patches
        self.vit.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.vit.pos_embed, std=0.02)
        
        # Replace classifier head
        self.vit.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize custom modules
        self.scorer = PatchImportanceScorer(patch_size=patch_size)
        self.selector = TopKPatchSelector(patch_percentage=patch_percentage)
        
        # Store metadata
        self.num_patches = num_patches
        self.embed_dim = embed_dim
    
    def forward(self, magno_image, line_drawing):
        """
        Forward pass through the model.
        
        Args:
            magno_image (torch.Tensor): Batch of Magno images (B, 3, H, W).
            line_drawing (torch.Tensor): Batch of line drawings (B, 1, H, W).
        
        Returns:
            torch.Tensor: Classification logits (B, num_classes).
        """
        # Score patches based on line drawing
        patch_scores = self.scorer(line_drawing)
        
        # Extract all patches from Magno image
        all_patches = self.vit.patch_embed(magno_image)
        
        # Select top-k patches with positional embeddings
        selected_patches = self.selector(
            all_patches,
            self.vit.pos_embed,
            patch_scores
        )
        
        # Prepare [CLS] token
        cls_token_with_pos = self.vit.cls_token + self.vit.pos_embed[:, :1, :]
        
        # Combine [CLS] token with selected patches
        batch_size = magno_image.shape[0]
        full_sequence = torch.cat([
            cls_token_with_pos.expand(batch_size, -1, -1),
            selected_patches
        ], dim=1)
        
        # Apply dropout
        full_sequence = self.vit.pos_drop(full_sequence)
        
        # Process through transformer blocks
        x = self.vit.blocks(full_sequence)
        x = self.vit.norm(x)
        
        # Extract [CLS] token output
        cls_output = x[:, 0]
        
        # Final classification
        logits = self.vit.head(cls_output)
        
        return logits
    
    def get_selected_patches_indices(self, line_drawing):
        """
        Get indices of selected patches for visualization.
        
        Args:
            line_drawing (torch.Tensor): Line drawing (B, 1, H, W).
        
        Returns:
            torch.Tensor: Indices of selected patches (B, k).
        """
        patch_scores = self.scorer(line_drawing)
        k = int(patch_scores.shape[1] * self.patch_percentage)
        _, indices = torch.topk(patch_scores, k=k, dim=1)
        return indices