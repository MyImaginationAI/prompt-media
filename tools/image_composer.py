#!/usr/bin/env python3

import os
from PIL import Image
import glob
import argparse
from typing import List, Tuple
import math

def load_images(directory: str) -> List[Image.Image]:
    """Load all images from a directory."""
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    images = []
    for img_path in sorted(image_files):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return images

def create_gif(images: List[Image.Image], output_path: str, duration: int = 500):
    """Create a GIF from a list of images."""
    if not images:
        print("No images found to create GIF")
        return

    # Use dimensions from the first image
    first_image = images[0]
    size = first_image.size
    print(f"Creating GIF with dimensions: {size[0]}x{size[1]}")
    
    # Standardize size for all images
    resized_images = [img.copy().resize(size, Image.Resampling.LANCZOS) for img in images]
    
    # Save GIF
    resized_images[0].save(
        output_path,
        save_all=True,
        append_images=resized_images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF created at: {output_path}")

def create_collage(images: List[Image.Image], output_path: str, max_width: int = 2400, 
                  border_size: int = 2, row_spacing: int = None):
    """Create a collage from a list of images.
    
    Args:
        images: List of images to include in collage
        output_path: Where to save the collage
        max_width: Maximum width of the collage
        border_size: Size of horizontal border between images in pixels
        row_spacing: Size of vertical space between rows in pixels (defaults to border_size if None)
    """
    if not images:
        print("No images found to create collage")
        return

    # If row_spacing is not specified, use border_size
    row_spacing = border_size if row_spacing is None else row_spacing

    # For wide images, we'll use 5 images per row
    images_per_row = 5
    num_rows = math.ceil(len(images) / images_per_row)

    # Get dimensions of first image to use as reference
    ref_width, ref_height = images[0].size
    aspect_ratio = ref_width / ref_height

    # Calculate image width based on max_width and borders
    image_width = (max_width - (images_per_row - 1) * border_size) // images_per_row
    image_height = int(image_width / aspect_ratio)

    # Calculate total dimensions
    total_width = (images_per_row * image_width) + ((images_per_row - 1) * border_size)
    total_height = (num_rows * image_height) + ((num_rows - 1) * row_spacing)

    # Create the collage
    collage = Image.new('RGB', (total_width, total_height), 'black')

    # Paste images
    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row

        # Calculate position
        x = col * (image_width + border_size)
        y = row * (image_height + row_spacing)

        # Resize image maintaining aspect ratio
        resized_img = img.copy()
        resized_img = resized_img.resize((image_width, image_height), Image.Resampling.LANCZOS)
        
        # Paste the image
        collage.paste(resized_img, (x, y))

    collage.save(output_path, quality=95)
    print(f"Collage created at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create GIF and collage from images in a directory')
    parser.add_argument('directory', help='Directory containing the images')
    parser.add_argument('--gif-output', help='Output path for GIF')
    parser.add_argument('--collage-output', help='Output path for collage')
    parser.add_argument('--gif-duration', type=int, help='Duration for each frame in GIF (ms)', default=500)
    parser.add_argument('--collage-width', type=int, help='Maximum width of collage', default=2400)
    parser.add_argument('--border-size', type=int, help='Size of horizontal border between images in pixels', default=2)
    parser.add_argument('--row-spacing', type=int, help='Size of vertical space between rows in pixels (defaults to border-size if not specified)')
    
    args = parser.parse_args()
    
    # Ensure directory exists
    if not os.path.exists(args.directory):
        print(f"Directory not found: {args.directory}")
        return

    # Load images only if we need to process them
    if not args.gif_output and not args.collage_output:
        print("No output specified. Use --gif-output and/or --collage-output")
        return

    # Load images
    images = load_images(args.directory)
    if not images:
        print("No images found in the specified directory")
        return

    # Create outputs based on what was requested
    if args.gif_output:
        create_gif(images, args.gif_output, args.gif_duration)
    if args.collage_output:
        create_collage(images, args.collage_output, args.collage_width, args.border_size, args.row_spacing)

if __name__ == '__main__':
    main()
