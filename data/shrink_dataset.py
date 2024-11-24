import argparse
import cv2
import glob
import os
from concurrent.futures import ThreadPoolExecutor

def resize_image(file_path, new_path, interpolation):
    """
    Resize the image to (1024, 768) using the specified interpolation.
    """
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not read {file_path}")
        return
    
    # Resize the image
    resized_image = cv2.resize(image, (1024, 768), interpolation=interpolation)
    
    # Determine the new file path
    relative_path = os.path.relpath(file_path, args.original_data_path)
    new_file_path = os.path.join(new_path, relative_path)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    
    # Save the resized image
    cv2.imwrite(new_file_path, resized_image)
    print(f"Resized and saved {new_file_path}")

def process_image(file_path):
    """
    Determine the correct interpolation and resize the image accordingly.
    """
    if file_path.endswith("_lab.png"):
        # Use nearest neighbor for label images
        resize_image(file_path, args.new_path, cv2.INTER_NEAREST)
    elif file_path.endswith(".jpg"):
        # Use default interpolation for .jpg images
        resize_image(file_path, args.new_path, cv2.INTER_LINEAR)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("original_data_path", help="Path of the original dataset")
    parser.add_argument("new_path", help="Path for the resized dataset")
    global args #That way it can be accessed directly in process_image and resize_image
    args = parser.parse_args()

    # Get all relevant image files in the directory (both _lab.png and .jpg)
    image_files = glob.glob(os.path.join(args.original_data_path, "**", "*.png"), recursive=True) + \
                  glob.glob(os.path.join(args.original_data_path, "**", "*.jpg"), recursive=True)
    
    #Filter files to remove where ColorMasks is present, cuz those don't need to be resized

    image_files = [image_file for image_file in image_files if "ColorMasks" not in image_file]


    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        executor.map(process_image, image_files)

if __name__ == "__main__":
    main()
