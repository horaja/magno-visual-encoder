import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2


def apply_log_filter_opencv(image_array, sigma=1.0, threshold=0.02):
    """
    Apply Laplacian of Gaussian filter using OpenCV (Method 2 from Medium article).
    """
    # Use consistent float64 throughout
    img_float = image_array.astype(np.float64) / 255.0
    
    # Step 1: Apply Gaussian blur
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaussian_blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
    
    # Step 2: Apply Laplacian (using CV_64F consistently)
    laplacian = cv2.Laplacian(gaussian_blurred, cv2.CV_64F)
    
    # Step 3: Detect zero crossings
    # A zero crossing occurs when the sign changes
    rows, cols = laplacian.shape
    zero_crossings = np.zeros((rows, cols), dtype=np.uint8)
    
    # Check horizontal neighbors
    for i in range(rows):
        for j in range(cols - 1):
            if laplacian[i, j] * laplacian[i, j + 1] < 0:
                if abs(laplacian[i, j] - laplacian[i, j + 1]) > threshold:
                    zero_crossings[i, j] = 255
    
    # Check vertical neighbors
    for i in range(rows - 1):
        for j in range(cols):
            if laplacian[i, j] * laplacian[i + 1, j] < 0:
                if abs(laplacian[i, j] - laplacian[i + 1, j]) > threshold:
                    zero_crossings[i, j] = 255
    
    # Check diagonal neighbors for more complete edges
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Top-left to bottom-right diagonal
            if laplacian[i, j] * laplacian[i + 1, j + 1] < 0:
                if abs(laplacian[i, j] - laplacian[i + 1, j + 1]) > threshold:
                    zero_crossings[i, j] = 255
            # Top-right to bottom-left diagonal
            if laplacian[i, j + 1] * laplacian[i + 1, j] < 0:
                if abs(laplacian[i, j + 1] - laplacian[i + 1, j]) > threshold:
                    zero_crossings[i, j + 1] = 255
    
    return zero_crossings


def apply_log_filter_opencv_simplified(image_array, sigma=1.0, threshold_percentile=98):
    """
    Simplified OpenCV LoG with percentile-based thresholding for sparsity control.
    """
    # Use consistent data type
    img_float = image_array.astype(np.float64)
    
    # Apply Gaussian blur
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
    
    # Apply Laplacian (using 64F consistently)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Get absolute values for edge strength
    edge_strength = np.abs(laplacian)
    
    # Threshold based on percentile to control sparsity
    threshold = np.percentile(edge_strength, threshold_percentile)
    
    # Create binary image
    binary_edges = np.zeros_like(image_array, dtype=np.uint8)
    binary_edges[edge_strength > threshold] = 255
    
    # Optional: Apply morphological operations to clean up
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel)
    
    return binary_edges


def apply_log_filter_marr_hildreth(image_array, sigma=1.0, threshold=0.01):
    """
    Marr-Hildreth edge detection (LoG with zero-crossing detection).
    More sophisticated implementation for cleaner edges.
    
    Args:
        image_array: Numpy array of the image
        sigma: Standard deviation for Gaussian kernel
        threshold: Threshold for zero-crossing detection
    
    Returns:
        Binary edge image
    """
    # Use consistent data type throughout (float64)
    img_float = image_array.astype(np.float64)
    
    # Apply Gaussian blur
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    # Make sure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
    
    # Apply Laplacian (using CV_64F consistently)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Find zero crossings with threshold
    rows, cols = laplacian.shape
    edges = np.zeros((rows, cols), dtype=np.uint8)
    
    # Use a 3x3 window to check for zero crossings
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Get 3x3 neighborhood
            neighborhood = laplacian[i-1:i+2, j-1:j+2]
            
            # Check if center pixel is close to zero
            if abs(laplacian[i, j]) < threshold:
                # Check for sign change in neighborhood
                pos = np.sum(neighborhood > 0)
                neg = np.sum(neighborhood < 0)
                
                # If we have both positive and negative values, it's a zero crossing
                if pos > 0 and neg > 0:
                    # Check if the gradient magnitude is significant
                    max_val = np.max(neighborhood)
                    min_val = np.min(neighborhood)
                    if (max_val - min_val) > threshold:
                        edges[i, j] = 255
    
    return edges


def process_image(input_path, output_path, target_size=224, sigma=1.0, 
                 method='simplified', threshold=98):
    """
    Process a single image with LoG filter using OpenCV.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        target_size: Size to resize image to (square)
        sigma: Standard deviation for LoG filter
        method: 'simplified', 'zero_cross', or 'marr_hildreth'
        threshold: Threshold value (percentile for simplified, absolute for others)
    """
    # Load image and convert to grayscale
    img = Image.open(input_path).convert('L')
    
    # Resize to target size
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Apply selected LoG method
    if method == 'simplified':
        result_array = apply_log_filter_opencv_simplified(img_array, sigma, threshold)
    elif method == 'zero_cross':
        result_array = apply_log_filter_opencv(img_array, sigma, threshold/100.0)
    elif method == 'marr_hildreth':
        result_array = apply_log_filter_marr_hildreth(img_array, sigma, threshold/100.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save the result
    result = Image.fromarray(result_array)
    result.save(output_path)


def process_dataset(input_dir, output_dir, target_size=224, sigma=1.0,
                   method='simplified', threshold=98):
    """
    Process all images in a dataset directory.
    
    Args:
        input_dir: Root directory containing class folders with images
        output_dir: Output directory to save processed images
        target_size: Size to resize images to
        sigma: Standard deviation for LoG filter
        method: LoG method to use
        threshold: Threshold value
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPEG'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Configuration:")
    print(f"  - Size: {target_size}x{target_size}")
    print(f"  - Sigma: {sigma}")
    print(f"  - Method: {method}")
    print(f"  - Threshold: {threshold}")
    
    # Track statistics
    processed = 0
    failed = 0
    total_white_pixels = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Construct relative path
        rel_path = img_path.relative_to(input_path)
        
        # Create output path maintaining directory structure
        out_path = output_path / rel_path
        
        # Change extension to .png and add suffix
        out_path = out_path.with_name(out_path.stem + '_log.png')
        
        # Create output directory if it doesn't exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            process_image(img_path, out_path, target_size, sigma, method, threshold)
            
            # Calculate sparsity statistics
            processed_img = np.array(Image.open(out_path))
            white_pixel_ratio = np.sum(processed_img > 0) / processed_img.size
            total_white_pixels += white_pixel_ratio
            
            processed += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            failed += 1
            continue
    
    # Print summary
    avg_white_ratio = (total_white_pixels / processed * 100) if processed > 0 else 0
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed} images")
    if failed > 0:
        print(f"  Failed: {failed} images")
    print(f"  Average white pixel ratio: {avg_white_ratio:.2f}%")
    print(f"  Output saved to: {output_dir}")


def test_parameters(input_image, output_dir, sigmas=[0.5, 1.0, 1.5, 2.0], 
                   thresholds=[95, 97, 98, 99]):
    """
    Test different parameter combinations on a single image.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    img = Image.open(input_image).convert('L')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    print(f"Testing parameters on {input_image}")
    print("=" * 50)
    
    for method in ['simplified', 'zero_cross', 'marr_hildreth']:
        print(f"\nMethod: {method}")
        print("-" * 30)
        
        for sigma in sigmas:
            for threshold in thresholds:
                if method == 'simplified':
                    result = apply_log_filter_opencv_simplified(img_array, sigma, threshold)
                elif method == 'zero_cross':
                    result = apply_log_filter_opencv(img_array, sigma, threshold/100.0)
                else:
                    result = apply_log_filter_marr_hildreth(img_array, sigma, threshold/100.0)
                
                white_ratio = np.sum(result > 0) / result.size * 100
                
                output_file = output_path / f"{method}_sigma{sigma}_thresh{threshold}.png"
                Image.fromarray(result).save(output_file)
                
                print(f"  σ={sigma}, τ={threshold}: {white_ratio:.2f}% white pixels")


def verify_output(output_dir):
    """Verify the processed dataset structure and report statistics."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Output directory does not exist: {output_dir}")
        return
    
    # Count images per class and check sparsity
    class_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    if class_dirs:
        print(f"\nDataset statistics for {output_path.name}:")
        print("-" * 40)
        total_images = 0
        total_white_ratios = []
        
        for class_dir in sorted(class_dirs):
            images = list(class_dir.glob('*.png'))
            num_images = len(images)
            total_images += num_images
            
            # Sample a few images for sparsity check
            sample_size = min(5, num_images)
            if images and sample_size > 0:
                class_white_ratios = []
                for img_path in images[:sample_size]:
                    img = np.array(Image.open(img_path))
                    white_ratio = np.sum(img > 0) / img.size * 100
                    class_white_ratios.append(white_ratio)
                
                avg_class_white = np.mean(class_white_ratios)
                total_white_ratios.extend(class_white_ratios)
                print(f"  {class_dir.name}: {num_images} images, ~{avg_class_white:.2f}% white pixels")
            else:
                print(f"  {class_dir.name}: {num_images} images")
        
        if total_white_ratios:
            overall_avg = np.mean(total_white_ratios)
            overall_std = np.std(total_white_ratios)
            print(f"\n  Total: {total_images} images")
            print(f"  Overall white pixels: {overall_avg:.2f}% ± {overall_std:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Apply Laplacian of Gaussian filter using OpenCV"
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing raw images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed images'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Target size for square images (default: 224)'
    )
    
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='Standard deviation for Gaussian kernel (default: 1.0, try 0.5-2.0)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=98,
        help='Threshold: percentile for simplified (90-99), absolute for others (0.01-0.1)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='simplified',
        choices=['simplified', 'zero_cross', 'marr_hildreth'],
        help='LoG method: simplified (fast), zero_cross (accurate), marr_hildreth (clean)'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        help='Test parameters on a single image'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify output after processing'
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode - try different parameters
        test_parameters(args.test, args.output_dir)
    else:
        # Process the dataset
        process_dataset(args.input_dir, args.output_dir, args.size, 
                       args.sigma, args.method, args.threshold)
        
        # Optionally verify output
        if args.verify:
            verify_output(args.output_dir)


if __name__ == '__main__':
    main()