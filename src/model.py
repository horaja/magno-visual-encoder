import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

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


class FlexiblePatchSelector(nn.Module):
    """
    Flexible patch selector supporting multiple selection strategies:
    - 'top_k': Select top-k patches based on scores
    - 'random': Randomly select patches
    - 'spatial_threshold': Spatially-biased threshold selection (replacing probabilistic)
    - 'threshold': Select patches above a threshold
    """
    def __init__(self, patch_percentage, selection_strategy='top_k'):
        """
        Args:
            patch_percentage (float): The fraction of patches to select (e.g., 0.25 for 25%).
            selection_strategy (str): Strategy for patch selection.
        """
        super().__init__()
        if not (0 < patch_percentage <= 1.0):
            raise ValueError("patch_percentage must be between 0 and 1.")
        self.patch_percentage = patch_percentage
        self.selection_strategy = selection_strategy
        
        # Parameters for spatial threshold strategy
        self.threshold = 0.3  # Can be made configurable
        self.gaussian_std = 0.25
        self.use_center_of_gravity = True
        self.normalize_scores = True

    def compute_center_of_gravity(self, line_drawing):
        """
        Compute the center of gravity (center of mass) of the line drawing.
        
        Args:
            line_drawing (torch.Tensor): Shape (B, 1, H, W)
        
        Returns:
            torch.Tensor: Centers of gravity, shape (B, 2) with (y, x) coordinates normalized to [0, 1]
        """
        B, _, H, W = line_drawing.shape
        
        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=line_drawing.device).view(1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=line_drawing.device).view(1, 1, 1, W)
        
        # Compute center of gravity for each image in batch
        total_mass = line_drawing.sum(dim=(2, 3), keepdim=True) + 1e-6  # Avoid division by zero
        
        cog_y = (line_drawing * y_coords).sum(dim=(2, 3)).squeeze(-1) / total_mass.squeeze()
        cog_x = (line_drawing * x_coords).sum(dim=(2, 3)).squeeze(-1) / total_mass.squeeze()
        
        return torch.stack([cog_y, cog_x], dim=1)
    
    def create_gaussian_weights(self, num_patches_h, num_patches_w, centers, device):
        """
        Create 2D Gaussian weights for spatial biasing.
        
        Args:
            num_patches_h (int): Number of patches in height
            num_patches_w (int): Number of patches in width
            centers (torch.Tensor): Centers for Gaussian, shape (B, 2)
            device: Device to create tensor on
        
        Returns:
            torch.Tensor: Gaussian weights, shape (B, num_patches_h * num_patches_w)
        """
        B = centers.shape[0]
        
        # Create patch coordinate grid (normalized to [0, 1])
        y_patch = torch.linspace(0, 1, num_patches_h, device=device)
        x_patch = torch.linspace(0, 1, num_patches_w, device=device)
        grid_y, grid_x = torch.meshgrid(y_patch, x_patch, indexing='ij')
        
        # Flatten grid coordinates
        grid_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # (N_patches, 2)
        
        # Compute Gaussian weights for each batch item
        weights = []
        for b in range(B):
            center = centers[b].unsqueeze(0)  # (1, 2)
            
            # Compute distances from center
            distances_sq = ((grid_coords - center) ** 2).sum(dim=1)  # (N_patches,)
            
            # Apply Gaussian
            gaussian_weights = torch.exp(-distances_sq / (2 * self.gaussian_std ** 2))
            weights.append(gaussian_weights)
        
        return torch.stack(weights)  # (B, N_patches)

    def forward(self, magno_patches, vit_positional_embedding, scores=None, line_drawing=None):
        """
        Args:
            magno_patches (torch.Tensor): All patches from the Magno image,
                                          shape (B, num_patches, patch_dim).
            vit_positional_embedding (torch.Tensor): The ViT's standard positional
                                                     embeddings, shape (1, num_patches + 1, embed_dim).
            scores (torch.Tensor, optional): The density scores from the PatchImportanceScorer,
                                            shape (B, num_patches). Not used for random selection.
            line_drawing (torch.Tensor, optional): Original line drawing for spatial_threshold,
                                                   shape (B, 1, H, W).
        
        Returns:
            torch.Tensor: A sparse sequence of selected patch embeddings with their
                          positional encodings added, shape (B, num_selected_patches, embed_dim).
        """
        B, N, D = magno_patches.shape
        k = int(N * self.patch_percentage)

        # Get indices based on selection strategy
        if self.selection_strategy == 'random':
            # Random selection (ignore scores)
            indices = self._random_selection(B, N, k)
            
        elif self.selection_strategy == 'top_k':
            # Original top-k selection
            if scores is None:
                raise ValueError("Scores required for top_k selection")
            _, indices = torch.topk(scores, k=k, dim=1)
            
        elif self.selection_strategy == 'spatial_threshold':
            # NEW: Spatially-biased threshold selection (replacing probabilistic)
            if scores is None or line_drawing is None:
                raise ValueError("Scores and line_drawing required for spatial_threshold selection")
            indices = self._spatial_threshold_selection(scores, line_drawing, k)
            
        elif self.selection_strategy == 'threshold':
            # Threshold-based selection with top-k fallback
            if scores is None:
                raise ValueError("Scores required for threshold selection")
            indices = self._threshold_selection(scores, k)
            
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        # Gather the selected patches using the indices
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        selected_patches = torch.gather(magno_patches, 1, indices_expanded)

        # Get positional embeddings for selected patches
        pos_embed_patches = vit_positional_embedding[:, 1:, :]  # Skip CLS token
        pos_embed_indices = indices.unsqueeze(-1).expand(-1, -1, pos_embed_patches.size(-1))
        selected_pos_embed = torch.gather(pos_embed_patches.expand(B, -1, -1), 1, pos_embed_indices)

        # Add positional embeddings to selected patches
        return selected_patches + selected_pos_embed

    def _random_selection(self, batch_size, num_patches, k):
        """Randomly select k patches for each sample in batch."""
        indices = torch.zeros(batch_size, k, dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            indices[i] = torch.randperm(num_patches)[:k]
        return indices

    def _spatial_threshold_selection(self, scores, line_drawing, k):
        """
        Select patches using spatially-biased threshold method.
        
        Args:
            scores (torch.Tensor): Importance scores, shape (B, num_patches)
            line_drawing (torch.Tensor): Line drawings, shape (B, 1, H, W)
            k (int): Target number of patches to select
        
        Returns:
            torch.Tensor: Selected indices, shape (B, k)
        """
        B, N = scores.shape
        device = scores.device
        
        # Calculate patch grid dimensions
        num_patches_per_side = int(np.sqrt(N))
        
        # Step 1: Compute center of gravity
        if self.use_center_of_gravity:
            centers = self.compute_center_of_gravity(line_drawing)
        else:
            # Use image center
            centers = torch.tensor([[0.5, 0.5]], device=device).repeat(B, 1)
        
        # Step 2: Create Gaussian spatial weights
        gaussian_weights = self.create_gaussian_weights(
            num_patches_per_side, num_patches_per_side, centers, device
        )
        
        # Step 3: Combine scores with spatial weights
        weighted_scores = scores * gaussian_weights
        
        # Step 4: Normalize scores if requested
        if self.normalize_scores:
            weighted_scores = weighted_scores / (weighted_scores.sum(dim=1, keepdim=True) + 1e-6)
        
        # Step 5: Apply threshold selection with fallback to top-k
        indices_list = []
        for b in range(B):
            # Get patches above threshold
            above_threshold = weighted_scores[b] > self.threshold
            above_indices = torch.where(above_threshold)[0]
            
            if len(above_indices) >= k:
                # Too many patches above threshold, take top-k
                _, top_indices = torch.topk(weighted_scores[b], k=k)
                indices_list.append(top_indices)
            elif len(above_indices) > 0:
                # Some but not enough patches above threshold
                # Take all above threshold + top remaining
                remaining_k = k - len(above_indices)
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[above_indices] = False
                remaining_scores = weighted_scores[b].clone()
                remaining_scores[~mask] = -float('inf')
                _, remaining_indices = torch.topk(remaining_scores, k=remaining_k)
                combined = torch.cat([above_indices, remaining_indices])
                indices_list.append(combined[:k])
            else:
                # No patches above threshold, fall back to top-k
                _, top_indices = torch.topk(weighted_scores[b], k=k)
                indices_list.append(top_indices)
        
        return torch.stack(indices_list)

    def _threshold_selection(self, scores, k):
        """Select patches above adaptive threshold, with top-k fallback."""
        B, N = scores.shape
        indices_list = []
        
        for i in range(B):
            # Compute adaptive threshold (e.g., mean + 0.5 * std)
            score_mean = scores[i].mean()
            score_std = scores[i].std()
            threshold = score_mean + 0.5 * score_std
            
            # Get patches above threshold
            above_threshold = scores[i] > threshold
            above_indices = torch.where(above_threshold)[0]
            
            if len(above_indices) >= k:
                # If too many patches above threshold, take top-k
                _, top_indices = torch.topk(scores[i], k=k)
                indices_list.append(top_indices)
            elif len(above_indices) > 0:
                # If some but not enough patches above threshold, 
                # take all above threshold + top remaining
                remaining_k = k - len(above_indices)
                mask = torch.ones(N, dtype=torch.bool, device=scores.device)
                mask[above_indices] = False
                remaining_scores = scores[i].clone()
                remaining_scores[~mask] = -float('inf')
                _, remaining_indices = torch.topk(remaining_scores, k=remaining_k)
                combined = torch.cat([above_indices, remaining_indices])
                indices_list.append(combined[:k])
            else:
                # If no patches above threshold, fall back to top-k
                _, top_indices = torch.topk(scores[i], k=k)
                indices_list.append(top_indices)
        
        return torch.stack(indices_list)


# Keep the old TopKPatchSelector for backward compatibility
class TopKPatchSelector(FlexiblePatchSelector):
    """Legacy class for backward compatibility."""
    def __init__(self, patch_percentage):
        super().__init__(patch_percentage, selection_strategy='top_k')


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
        selection_strategy (str): Strategy for patch selection ('top_k', 'random', 'probabilistic', 'threshold').
    """
    
    def __init__(self, 
                 patch_percentage=0.25,
                 num_classes=10,
                 img_size=64,
                 patch_size=4,
                 vit_model_name='vit_tiny_patch16_224.augreg_in21k',
                 embed_dim=None,
                 selection_strategy='top_k'):
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
        self.selection_strategy = selection_strategy
        
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
        self.selector = FlexiblePatchSelector(
            patch_percentage=patch_percentage,
            selection_strategy=selection_strategy
        )
        
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
        # Score patches based on line drawing (even for random, we compute for analysis)
        patch_scores = self.scorer(line_drawing) if line_drawing is not None else None
        
        # Extract all patches from Magno image
        all_patches = self.vit.patch_embed(magno_image)
        
        # Select patches based on strategy
        # Pass line_drawing for spatial_threshold strategy
        selected_patches = self.selector(
            all_patches,
            self.vit.pos_embed,
            patch_scores,
            line_drawing if self.selection_strategy == 'spatial_threshold' else None
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
    
    def set_patch_percentage(self, new_percentage):
        """Dynamically adjust patch percentage."""
        if not 0 < new_percentage <= 1.0:
            raise ValueError(f"patch_percentage must be in (0, 1], got {new_percentage}")
        
        self.patch_percentage = new_percentage
        self.selector.patch_percentage = new_percentage
        return self
    
    def set_selection_strategy(self, strategy):
        """Change the patch selection strategy."""
        valid_strategies = ['top_k', 'random', 'probabilistic', 'threshold']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got {strategy}")
        
        self.selection_strategy = strategy
        self.selector.selection_strategy = strategy
        return self
        
    @torch.no_grad()
    def get_selected_patches_indices(self, line_drawing, patch_percentage=None):
        """Get indices of selected patches for visualization."""
        if self.selection_strategy == 'random':
            # For random, just return random indices
            B = line_drawing.shape[0]
            N = self.num_patches
            k = int(N * (patch_percentage or self.patch_percentage))
            return torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        else:
            patch_scores = self.scorer(line_drawing)
            k = int(patch_scores.shape[1] * (patch_percentage or self.patch_percentage))
            
            if self.selection_strategy == 'top_k':
                _, indices = torch.topk(patch_scores, k=k, dim=1)
            elif self.selection_strategy == 'spatial_threshold':
                indices = self.selector._spatial_threshold_selection(patch_scores, line_drawing, k)
            elif self.selection_strategy == 'threshold':
                indices = self.selector._threshold_selection(patch_scores, k)
            
            return indices 
        
    def get_model_info(self):
        """
        Get model configuration information.
        
        Returns:
            dict: Model configuration details.
        """
        return {
            'patch_percentage': self.patch_percentage,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'num_patches': self.num_patches,
            'embed_dim': self.embed_dim,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }