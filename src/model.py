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
    The complete SelectiveMagnoViT model.
    """
    def __init__(self, patch_percentage=0.25, num_classes=10, vit_model_name='vit_base_patch16_224_in21k'):
        """
        Args:
            patch_percentage (float): The percentage of patches to select.
            num_classes (int): The number of output classes.
            vit_model_name (str): The name of the pre-trained ViT model from timm.
        """
        super().__init__()
        
        # --- Load the ViT Backbone (Module 5) ---
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.patch_size = self.vit.patch_embed.patch_size[0]
        
        # --- Instantiate Our Custom Modules ---
        self.scorer = PatchImportanceScorer(patch_size=self.patch_size)
        self.selector = TopKPatchSelector(patch_percentage=patch_percentage)

        # --- Classifier Head ---
        # Replace the ViT's head with a new one for our number of classes
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, magno_image, line_drawing):
        """
        The forward pass implementing the full SelectiveMagnoViT pipeline.
        
        Args:
            magno_image (torch.Tensor): Batch of Magno images (B, 3, H, W).
            line_drawing (torch.Tensor): Batch of Line Drawings (B, 1, H, W).
            
        Returns:
            torch.Tensor: The final classification logits (B, num_classes).
        """
        # --- Module 2: Score Patches ---
        # Get density scores from the line drawing
        patch_scores = self.scorer(line_drawing)

        # --- Pre-computation for Module 3 & 4 ---
        # Use the ViT's patch embedding layer to get all patches from the Magno image
        all_magno_patches = self.vit.patch_embed(magno_image) # (B, num_patches, embed_dim)
        
        # Get the ViT's positional embeddings
        pos_embed = self.vit.pos_embed # (1, num_patches + 1, embed_dim)
        
        # --- Module 3 & 4: Select Top-K Patches and Add Positional Embeddings ---
        selected_patch_embeddings = self.selector(all_magno_patches, pos_embed, patch_scores)
        
        # --- Module 5: Process with ViT Backbone ---
        # Get the [CLS] token from the ViT and add its positional embedding
        cls_token_with_pos = self.vit.cls_token + pos_embed[:, :1, :] # (1, 1, D)
        
        # Prepend the [CLS] token to the sequence of selected patches
        # (B, k, D) -> (B, k + 1, D)
        full_sequence = torch.cat([cls_token_with_pos.expand(magno_image.shape[0], -1, -1), selected_patch_embeddings], dim=1)
        
        # Apply dropout
        full_sequence = self.vit.pos_drop(full_sequence)
        
        # Pass the sparse sequence through the transformer blocks
        x = self.vit.blocks(full_sequence)
        x = self.vit.norm(x)
        
        # Get the output corresponding to the [CLS] token for classification
        cls_output = x[:, 0]
        
        # Pass through the final classifier head
        logits = self.vit.head(cls_output)
        
        return logits

# if __name__ == '__main__':
#     # --- Verification and Example Usage ---
#     model = SelectiveMagnoViT(patch_percentage=0.25, num_classes=10).cuda()
#     model.eval()

#     # Create dummy input tensors
#     magno_img = torch.randn(4, 3, 224, 224).cuda()  # Batch of 4
#     line_img = torch.rand(4, 1, 224, 224).cuda()   # Values between 0 and 1
    
#     print("--- Model Instantiated ---")
#     print(f"Patch Size: {model.patch_size}")
    
#     # Run a forward pass
#     with torch.no_grad():
#         output_logits = model(magno_img, line_img)
    
#     print("\n--- Verification ---")
#     print(f"Input Magno Shape: {magno_img.shape}")
#     print(f"Input Line Shape:  {line_img.shape}")
#     print(f"Output Logits Shape: {output_logits.shape}")
    
#     # Check output shape
#     assert output_logits.shape == (4, 10), "Output shape is incorrect!"
#     print("\nModel forward pass successful!")