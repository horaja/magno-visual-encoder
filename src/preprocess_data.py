import os
import argparse
import cv2
from tqdm import tqdm

def magno_transform(image, size=(224, 224)):
  """
  Applies the Module 0 "Magno Transform" to an image.
  Converts to grayscale and resizes to a fixed low resolution.
  
  Args:
    image (np.array): The input image in BGR format from OpenCV.
    size (tuple): The target (width, height) for resizing.
    
  Returns:
    np.array: The transformed grayscale image.
  """
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Resize the image to the target low resolution
  resized_image = cv2.resize(gray_image, size, interpolation=cv2.INTER_AREA)
  
  return resized_image

def main(args):
  """
  Main function to run the Magno Transform pre-processing.
  This script prepares images for the external line drawing generator.
  """
  # Create the output directory if it doesn't exist
  os.makedirs(args.output_dir, exist_ok=True)
  print(f"Output will be saved to: {args.output_dir}")

  # Get a list of all image files in the input directory
  try:
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
      print(f"Error: No images found in {args.input_dir}")
      return
    print(f"Found {len(image_files)} images to process.")
  except FileNotFoundError:
    print(f"Error: Input directory not found at {args.input_dir}")
    return

  # --- Main Processing Loop ---
  for filename in tqdm(image_files, desc="Applying Magno Transform"):
    try:
      # Construct full paths for input and output
      input_path = os.path.join(args.input_dir, filename)
      base_filename = os.path.splitext(filename)[0]
      # Use a consistent naming scheme for the output files
      output_filename = f"{base_filename}_magno.png"
      output_path = os.path.join(args.output_dir, output_filename)

      # Read the raw image using OpenCV
      image = cv2.imread(input_path)
      if image is None:
        print(f"Warning: Could not read {input_path}. Skipping.")
        continue

      # --- Module 0: Apply the Magno Transform ---
      magno_image_np = magno_transform(image, size=(args.size, args.size))
      
      # Save the processed Magno-transformed image
      cv2.imwrite(output_path, magno_image_np)

    except Exception as e:
      print(f"Error processing {filename}: {e}")

  print("\nMagno Transform pre-processing complete.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Pre-process raw images into low-resolution, grayscale 'Magno' images.")
  parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing the raw input images.")
  parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the processed 'Magno' images will be saved.")
  parser.add_argument("--size", type=int, default=224, help="The target resolution (size x size) for the output images.")
  
  args = parser.parse_args()
  main(args)
