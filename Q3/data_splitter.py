import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# IMPORTANT: Update these paths to match your directory structure.
# This should be the single folder containing all your images and XMLs
BASE_DIR = 'C:/ALL/FindWork/Dhvani Hackathon Mohammed Salim/TrafficDataset/train/Final Train Dataset'
# This is the 'labels' folder the converter script created
LABELS_DIR = 'C:/ALL/FindWork/Dhvani Hackathon Mohammed Salim/TrafficDataset/train/labels'

# Define the output directories for the split data
OUTPUT_ROOT = os.path.dirname(BASE_DIR)
TRAIN_IMG_DIR = os.path.join(OUTPUT_ROOT, 'images', 'train')
VAL_IMG_DIR = os.path.join(OUTPUT_ROOT, 'images', 'val')
TRAIN_LABEL_DIR = os.path.join(OUTPUT_ROOT, 'labels', 'train')
VAL_LABEL_DIR = os.path.join(OUTPUT_ROOT, 'labels', 'val')

def split_dataset():
    """
    Splits the dataset into training and validation sets and organizes the files.
    """
    print("Starting data splitting...")

    # Get the list of all image files
    all_images = [f for f in os.listdir(BASE_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    
    if not all_images:
        print(f"No images found in {BASE_DIR}. Please check the path.")
        return

    # Split the image filenames into training and validation sets
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    # Create the output directories if they don't exist
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)

    print(f"Splitting {len(all_images)} images into {len(train_images)} training and {len(val_images)} validation files...")

    # Copy files to their new training and validation folders
    for image_name in train_images:
        label_name = os.path.splitext(image_name)[0] + '.txt'
        
        # Check if the label file exists before copying
        if not os.path.exists(os.path.join(LABELS_DIR, label_name)):
            print(f"Warning: Label file for {image_name} not found. Skipping.")
            continue

        shutil.copy(os.path.join(BASE_DIR, image_name), TRAIN_IMG_DIR)
        shutil.copy(os.path.join(LABELS_DIR, label_name), TRAIN_LABEL_DIR)

    for image_name in val_images:
        label_name = os.path.splitext(image_name)[0] + '.txt'
        
        if not os.path.exists(os.path.join(LABELS_DIR, label_name)):
            print(f"Warning: Label file for {image_name} not found. Skipping.")
            continue
            
        shutil.copy(os.path.join(BASE_DIR, image_name), VAL_IMG_DIR)
        shutil.copy(os.path.join(LABELS_DIR, label_name), VAL_LABEL_DIR)

    print("\nData splitting and organization complete.")
    print(f"Training images are in: {TRAIN_IMG_DIR}")
    print(f"Validation images are in: {VAL_IMG_DIR}")
    print(f"Training labels are in: {TRAIN_LABEL_DIR}")
    print(f"Validation labels are in: {VAL_LABEL_DIR}")
    print("\nYou can now proceed with YOLO training.")

if __name__ == '__main__':
    split_dataset()
