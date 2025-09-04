import numpy as np

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

class SpatialThresholdSelector(nn.Module):
    """
    Module 2: Selects patches using a spatially-biased threshold mechanism.
    
    This selector combines patch density scores with a spatial bias derived
    from the line drawing's center of gravity. It selects patches that are
    both dense and centrally located relative to the drawing's content.
    """
    def __init__(self, patch_percentage):
        """
        Args:
            patch_percentage (float): The target fraction of patches to select.
        """
        super().__init__()
        if not (0 < patch_percentage <= 1.0):
            raise ValueError("patch_percentage must be between 0 and 1.")
        self.patch_percentage = patch_percentage
        
        # Parameters for the spatial threshold strategy
        self.threshold = 0.3
        self.gaussian_std = 0.25

    def _compute_center_of_gravity(self, line_drawing):
        """Computes the center of mass for each line drawing in the batch."""
        B, _, H, W = line_drawing.shape
        y_coords = torch.linspace(0, 1, H, device=line_drawing.device).view(1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=line_drawing.device).view(1, 1, 1, W)
        
        # CORRECTED SECTION: Ensure correct tensor shapes before division
        # Squeeze total_mass to prevent incorrect broadcasting. Shape becomes (B, 1)
        total_mass = line_drawing.sum(dim=(2, 3)) + 1e-6 

        # Summed coordinates also have shape (B, 1)
        sum_y = (line_drawing * y_coords).sum(dim=(2, 3))
        sum_x = (line_drawing * x_coords).sum(dim=(2, 3))

        cog_y = sum_y / total_mass # Shape (B, 1)
        cog_x = sum_x / total_mass # Shape (B, 1)
        
        # Stack and squeeze to get the final correct shape of (B, 2)
        return torch.stack([cog_y.squeeze(-1), cog_x.squeeze(-1)], dim=1)

    def _create_gaussian_weights(self, num_patches_h, num_patches_w, centers, device):
        """Creates a 2D Gaussian weight map centered for each batch item."""
        B = centers.shape[0]
        y_patch = torch.linspace(0, 1, num_patches_h, device=device)
        x_patch = torch.linspace(0, 1, num_patches_w, device=device)
        grid_y, grid_x = torch.meshgrid(y_patch, x_patch, indexing='ij')
        
        grid_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1) # (N_patches, 2)
        
        weights = []
        for b in range(B):
            center = centers[b].unsqueeze(0) # (1, 2)
            distances_sq = ((grid_coords - center) ** 2).sum(dim=1)
            gaussian_weights = torch.exp(-distances_sq / (2 * self.gaussian_std ** 2))
            weights.append(gaussian_weights)
        
        return torch.stack(weights) # (B, N_patches)
    
    def _get_selection_indices(self, scores, line_drawing, k):
        """Determines the indices of patches to select."""
        B, N = scores.shape
        device = scores.device
        num_patches_side = int(np.sqrt(N))

        # Step 1: Compute center of gravity and create spatial weights
        centers = self._compute_center_of_gravity(line_drawing)
        gaussian_weights = self._create_gaussian_weights(num_patches_side, num_patches_side, centers, device)

        # Step 2: Combine scores and apply thresholding with a fallback
        weighted_scores = scores * gaussian_weights
        
        indices_list = []
        for b in range(B):
            above_threshold_indices = torch.where(weighted_scores[b] > self.threshold)[0]
            
            if len(above_threshold_indices) >= k:
                # Too many patches passed, so take the top-k among them
                _, top_indices = torch.topk(weighted_scores[b], k=k)
                indices_list.append(top_indices)
            elif 0 < len(above_threshold_indices) < k:
                # Not enough patches, take them all and supplement with the next best
                remaining_k = k - len(above_threshold_indices)
                
                # Mask out already selected patches to find the next best
                remaining_scores = weighted_scores[b].clone()
                remaining_scores[above_threshold_indices] = -float('inf')
                
                _, remaining_indices = torch.topk(remaining_scores, k=remaining_k)
                combined_indices = torch.cat([above_threshold_indices, remaining_indices])
                indices_list.append(combined_indices)
            else:
                # No patches passed, so just fall back to top-k
                _, top_indices = torch.topk(weighted_scores[b], k=k)
                indices_list.append(top_indices)
                
        return torch.stack(indices_list)

    def forward(self, magno_patches, vit_positional_embedding, scores, line_drawing):
        """
        Selects and returns patches based on the spatial threshold strategy.
        """
        B, N, D = magno_patches.shape
        k = int(N * self.patch_percentage)

        # Get the indices of the selected patches
        selected_indices = self._get_selection_indices(scores, line_drawing, k)

        # Gather the selected patches using the indices
        indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_patches = torch.gather(magno_patches, 1, indices_expanded)

        # Gather the corresponding positional embeddings
        pos_embed_patches = vit_positional_embedding[:, 1:, :] # Skip CLS token
        pos_embed_indices = selected_indices.unsqueeze(-1).expand(-1, -1, pos_embed_patches.size(-1))
        selected_pos_embed = torch.gather(pos_embed_patches.expand(B, -1, -1), 1, pos_embed_indices)

        # Add positional embeddings to the selected patches
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
        self.selector = SpatialThresholdSelector(patch_percentage=patch_percentage)
        
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
        
        # Select patches using spatial threshold method with positional embeddings
        selected_patches = self.selector(
            all_patches,
            self.vit.pos_embed,
            patch_scores,
            line_drawing
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
    
    @torch.no_grad()
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