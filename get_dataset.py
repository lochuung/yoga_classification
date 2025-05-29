#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yoga Pose Dataset Downloader and Splitter Script
------------------------------------------------
This script downloads yoga pose images from URLs and organizes them into 
train/test datasets for training the yoga pose classification model.

Features:
- Downloads images from URLs in the Yoga-82 dataset (using multithreading for faster downloads)
- Processes images with MoveNet Thunder to extract pose keypoints
- Creates a structured directory for images
- Splits data into training and testing sets according to Yoga-82 splits
- Saves keypoint data to CSV files for training

Requirements:
- requests
- pillow
- opencv-python
- tqdm
- pandas
- tensorflow or tensorflow-lite

Usage:
  python get_dataset.py [--classes CLASSES] [--download] [--clean] [--no-keypoints]
  
  --classes        Comma-separated list of pose names to download
  --download       Force download from URLs even if images exist
  --clean          Remove existing downloaded images before downloading
  --no-keypoints   Skip keypoint detection (if you only want to download images)
  --max-threads    Maximum number of download threads (default: 10)
  
Example:
  python get_dataset.py --classes chair,cobra,tree --download
"""

import os
import argparse
import random
import shutil
import csv
import time
import json
import threading
import concurrent.futures
from urllib.parse import urlparse
from queue import Queue
from pathlib import Path

# Required third-party libraries
try:
    import requests
    import cv2
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    import pandas as pd
    import tensorflow as tf
    from movenet_thunder.data import BodyPart, person_from_keypoints_with_scores
except ImportError:
    print("Please install required packages: pip install requests pillow opencv-python tqdm pandas tensorflow")
    exit(1)

# Map of pose names in Yoga-82 dataset to our simplified pose names
POSE_NAME_MAP = {
    'Tree_Pose_or_Vrksasana_': 'tree_pose',
    'Mountain_Pose_or_Tadasana_': 'mountain_pose',
    'Cobra_Pose_or_Bhujangasana_': 'cobra_pose',
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_': 'downward_facing_dog',
    'Chair_Pose_or_Utkatasana_': 'chair',
    'Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_': 'shoulder_stand',
    'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_': 'triangle_pose',
    'Warrior_I_Pose_or_Virabhadrasana_I_': 'warriori',
    'Warrior_II_Pose_or_Virabhadrasana_II_': 'warrior_2',
    'Warrior_III_Pose_or_Virabhadrasana_III_': 'warrioriii',
    'Plank_Pose_or_Kumbhakasana_': 'plank_pose',
    'Child_Pose_or_Balasana_': 'childs_pose',
    'Cat_Cow_Pose_or_Marjaryasana_': 'cat_cow',
    'Corpse_Pose_or_Savasana_': 'corpse_pose'
}

# Reverse lookup to identify which Yoga-82 poses to use for each of our poses
REVERSE_POSE_MAP = {}
for yoga82_pose, our_pose in POSE_NAME_MAP.items():
    if our_pose not in REVERSE_POSE_MAP:
        REVERSE_POSE_MAP[our_pose] = []
    REVERSE_POSE_MAP[our_pose].append(yoga82_pose)

# Define the MoveNet model configuration
MOVENET_MODEL_PATH = 'movenet_thunder.tflite'

def load_movenet_model():
    """Load the MoveNet Thunder model for pose detection"""
    interpreter = tf.lite.Interpreter(model_path=MOVENET_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def detect_pose(interpreter, image_path, class_name=None, class_id=None):
    """
    Detect pose in an image using MoveNet Thunder
    
    Args:
        interpreter: TFLite interpreter for MoveNet
        image_path: Path to the image file
        class_name: Name of the pose class
        class_id: ID of the pose class
        
    Returns:
        Tuple of (filename, keypoints_with_scores, class_id, class_name)
        or None if detection fails
    """
    try:
        # Read the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image)
        
        # Convert to RGB if image has alpha channel
        if image.shape[-1] == 4:
            image = tf.image.rgb_to_grayscale(image[..., :3])
              # Resize to 256x256 which is the input size for MoveNet Thunder
        input_image = tf.image.resize(image, (256, 256))
        
        # MoveNet Thunder expects uint8 input (0-255)
        if input_image.dtype != tf.uint8:
            input_image = tf.cast(input_image, dtype=tf.uint8)
        
        # Add batch dimension
        input_image = tf.expand_dims(input_image, axis=0)
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index']).squeeze()
        
        # Get original image dimensions
        img_height, img_width = image.shape[0], image.shape[1]
        
        # Get the filename (basename) from the image path
        filename = os.path.basename(image_path)
        
        return (filename, keypoints_with_scores, img_width, img_height, class_id, class_name)
    except Exception as e:
        print(f"Error detecting pose in {image_path}: {e}")
        return None

def extract_keypoints_to_csv_format(detection_result):
    """
    Convert detection result to CSV row format
    
    Args:
        detection_result: Tuple from detect_pose function
        
    Returns:
        List representing a CSV row with keypoint data
    """
    if detection_result is None:
        return None
    
    filename, keypoints_with_scores, img_width, img_height, class_id, class_name = detection_result
    
    # Initialize the row with the filename
    row = [filename]
    
    # Add keypoints (x, y, score) for all body parts
    for i in range(17):  # 17 keypoints in MoveNet
        # Convert normalized coordinates to pixel values
        y, x, score = keypoints_with_scores[i]
        x_px = int(x * img_width)
        y_px = int(y * img_height)
        
        # Add to row
        row.extend([x_px, y_px, score])
    
    # Add class info
    row.append(class_id if class_id is not None else 0)
    row.append(class_name)
    
    return row

def write_keypoints_to_csv(detections, csv_path):
    """
    Write keypoint detections to CSV file
    
    Args:
        detections: List of detection results
        csv_path: Path to save the CSV file
    """
    # Define CSV header
    header = ['filename']
    body_parts = [part.name for part in BodyPart]
    
    for body_part in body_parts:
        header.extend([f'{body_part}_x', f'{body_part}_y', f'{body_part}_score'])
    
    header.extend(['class_no', 'class_name'])
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for detection in detections:
            row = extract_keypoints_to_csv_format(detection)
            if row:
                writer.writerow(row)
                
    print(f"Saved {len(detections)} pose keypoints to {csv_path}")

def parse_yoga82_split_files(pose_classes):
    """
    Parse the Yoga-82 train/test split files
    
    Args:
        pose_classes: List of pose classes we're interested in
        
    Returns:
        Dictionary with filenames mapped to 'train' or 'test'
    """
    split_map = {}
    
    for split_type in ['train', 'test']:
        split_file = os.path.join('yoga-82-dataset', f'yoga_{split_type}.txt')
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file {split_file} not found")
            continue
            
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                    
                file_path = parts[0]
                
                # Extract the pose name from the file path
                pose_name = file_path.split('/')[0]
                
                # Check if this pose maps to one of our pose classes
                matches_any = False
                for our_pose in pose_classes:
                    yoga82_poses = REVERSE_POSE_MAP.get(our_pose, [])
                    if pose_name in yoga82_poses or pose_name + '_' in yoga82_poses:
                        matches_any = True
                        # Add to split map with our pose name
                        split_map[file_path] = {
                            'split': split_type,
                            'original_pose': pose_name,
                            'our_pose': our_pose
                        }
                        break
    
    return split_map

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download and split yoga pose images')
    parser.add_argument('--classes', type=str, 
                        default='chair,cobra,dog,warrior,tree,traingle,shoudler_stand,no_pose',
                        help='Comma-separated list of pose classes to download')
    parser.add_argument('--download', action='store_true', 
                        help='Force download from URLs even if images exist')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of images to use for training (default: 0.8)')
    parser.add_argument('--clean', action='store_true',
                        help='Remove existing downloaded images before downloading')
    parser.add_argument('--max-per-pose', type=int, default=500,
                        help='Maximum number of images to download per pose class (default: 500)')
    parser.add_argument('--max-threads', type=int, default=10,
                        help='Maximum number of download threads (default: 10)')
    parser.add_argument('--no-keypoints', action='store_true',
                        help='Skip keypoint detection (just download and organize images)')
    parser.add_argument('--use-yoga82-split', action='store_true',
                        help='Use original Yoga-82 train/test split instead of random splitting')
    return parser.parse_args()

def download_image(url, save_path, timeout=10):
    """Download an image from a URL and save it to a path"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Skip if file already exists
        if os.path.exists(save_path):
            return True
        
        # Download the image
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            print(f"Failed to download {url}: HTTP {response.status_code}")
            return False
            
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        # Verify the image can be opened
        try:
            img = Image.open(save_path)
            img.verify()
            img.close()
            
            # Also check with OpenCV
            img = cv2.imread(save_path)
            if img is None or img.size == 0:
                raise Exception("Invalid image data")
                
            return True
        except Exception as e:
            print(f"Invalid image from {url}: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
            
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def download_pose_images(pose_name, max_images=500, force_download=False, max_threads=10):
    """Download images for a specific pose using multithreading for speed"""
    # Check for matching yoga-82 pose names
    yoga82_pose_names = REVERSE_POSE_MAP.get(pose_name, [])
    if not yoga82_pose_names:
        print(f"Warning: No mapping found for pose '{pose_name}' in the Yoga-82 dataset")
        return []
    
    downloaded_images = []
    total_downloaded = 0
    
    # Create a download queue
    download_tasks = []
    
    # Gather all download tasks
    for yoga82_pose in yoga82_pose_names:
        links_file = os.path.join('yoga-82-dataset', 'yoga_dataset_links', f"{yoga82_pose}.txt")
        
        # Skip if links file doesn't exist
        if not os.path.exists(links_file):
            print(f"Warning: Links file not found for {yoga82_pose}")
            continue
            
        # Read the links file
        with open(links_file, 'r') as f:
            lines = f.readlines()
            
        print(f"Adding download tasks for {yoga82_pose} -> {pose_name} ({len(lines)} URLs)")
        
        # Shuffle lines to get diverse samples
        random.shuffle(lines)
        
        # Restrict number of images per pose
        max_from_this_pose = max(50, int(max_images / len(yoga82_pose_names)))
        count = 0
        
        for line in lines:
            if count >= max_from_this_pose or total_downloaded >= max_images:
                break
                
            try:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                    
                file_path, url = parts
                
                # Extract filename from the path
                _, filename = os.path.split(file_path)
                
                # Create target path with our pose name
                downloads_dir = os.path.join('downloads', pose_name)
                save_path = os.path.join(downloads_dir, filename)
                
                # Check if file already exists
                if force_download or not os.path.exists(save_path):
                    download_tasks.append((url, save_path, force_download))
                    count += 1
                else:
                    # Image already exists
                    downloaded_images.append(save_path)
                    total_downloaded += 1
                    count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
    
    # Using ThreadPoolExecutor to download images in parallel
    if download_tasks:
        print(f"Downloading {len(download_tasks)} images using {max_threads} threads")
        
        # Create shared progress bar
        pbar = tqdm(total=len(download_tasks), desc=f"Downloading {pose_name}")
        
        # Thread-safe list for storing results
        results = []
        mutex = threading.Lock()
        
        def download_with_update(task):
            url, save_path, force = task
            success = download_image(url, save_path)
            
            # Update results and progress bar
            with mutex:
                if success:
                    results.append(save_path)
                pbar.update(1)
                
            return success
        
        # Execute downloads in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(download_with_update, task) for task in download_tasks]
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
        
        # Close progress bar
        pbar.close()
        
        # Add downloaded images to our list
        downloaded_images.extend(results)
        total_downloaded += len(results)
    
    print(f"Downloaded {total_downloaded} images for {pose_name}")
    return downloaded_images

def copy_existing_dataset(pose_classes):
    """Copy images from the existing yoga_poses folder to downloads folder"""
    existing_images = {}
    
    for split in ['train', 'test']:
        src_dir = os.path.join('yoga_poses', split)
        if not os.path.exists(src_dir):
            continue
            
        for pose in os.listdir(src_dir):
            if pose not in pose_classes:
                continue
                
            pose_dir = os.path.join(src_dir, pose)
            if not os.path.isdir(pose_dir):
                continue
                
            # Create the target directory
            target_dir = os.path.join('downloads', pose)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy each image
            if pose not in existing_images:
                existing_images[pose] = []
                
            for img_file in os.listdir(pose_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                src_path = os.path.join(pose_dir, img_file)
                # Use a special prefix to identify existing data
                target_name = f"existing_{split}_{img_file}"
                target_path = os.path.join(target_dir, target_name)
                
                # Copy the image if it doesn't already exist
                if not os.path.exists(target_path):
                    shutil.copy(src_path, target_path)
                
                existing_images[pose].append(target_path)
    
    return existing_images

def split_dataset(images_by_pose, train_ratio=0.8):
    """Split images into training and testing sets"""
    train_images = []
    test_images = []
    
    for pose, img_paths in images_by_pose.items():
        # Identify images that were already in train/test sets
        existing_train = [p for p in img_paths if 'existing_train_' in os.path.basename(p)]
        existing_test = [p for p in img_paths if 'existing_test_' in os.path.basename(p)]
        new_images = [p for p in img_paths if 'existing_' not in os.path.basename(p)]
        
        # Keep existing split
        train_images.extend([(p, pose) for p in existing_train])
        test_images.extend([(p, pose) for p in existing_test])
        
        # Split new images
        random.shuffle(new_images)
        split_idx = int(len(new_images) * train_ratio)
        
        train_images.extend([(p, pose) for p in new_images[:split_idx]])
        test_images.extend([(p, pose) for p in new_images[split_idx:]])
    
    return train_images, test_images

def save_splits_to_folders(train_images, test_images):
    """Save the split images to train/test folders"""
    # Create directories
    for split in ['train', 'test']:
        os.makedirs(os.path.join('yoga_poses', split), exist_ok=True)
    
    # Process training images
    print(f"Creating training set with {len(train_images)} images")
    for img_path, pose in tqdm(train_images, desc="Processing training images"):
        target_dir = os.path.join('yoga_poses', 'train', pose)
        os.makedirs(target_dir, exist_ok=True)
        
        # Get original filename, removing any 'existing_' prefix
        filename = os.path.basename(img_path)
        if filename.startswith('existing_train_') or filename.startswith('existing_test_'):
            filename = filename.split('_', 2)[2]
        
        target_path = os.path.join(target_dir, filename)
        
        # Copy image to target location
        if not os.path.exists(target_path):
            shutil.copy(img_path, target_path)
    
    # Process testing images
    print(f"Creating test set with {len(test_images)} images")
    for img_path, pose in tqdm(test_images, desc="Processing test images"):
        target_dir = os.path.join('yoga_poses', 'test', pose)
        os.makedirs(target_dir, exist_ok=True)
        
        # Get original filename, removing any 'existing_' prefix
        filename = os.path.basename(img_path)
        if filename.startswith('existing_train_') or filename.startswith('existing_test_'):
            filename = filename.split('_', 2)[2]
        
        target_path = os.path.join(target_dir, filename)
        
        # Copy image to target location
        if not os.path.exists(target_path):
            shutil.copy(img_path, target_path)

def save_class_csv_files(train_images, test_images):
    """Save CSV files with pose landmarks for each class"""
    # Create directory for CSVs
    os.makedirs('csv_per_pose', exist_ok=True)
    
    # Get unique poses
    all_poses = set([pose for _, pose in train_images + test_images])
    
    for pose in all_poses:
        # Get images for this pose
        pose_train_images = [img_path for img_path, img_pose in train_images if img_pose == pose]
        pose_test_images = [img_path for img_path, img_pose in test_images if img_pose == pose]
        all_pose_images = pose_train_images + pose_test_images
        
        print(f"Creating CSV file for {pose} ({len(all_pose_images)} images)")
        
        # Create a placeholder CSV
        csv_path = os.path.join('csv_per_pose', f"{pose}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'pose'])
            
            for img_path in all_pose_images:
                # Get relative path for the image
                rel_path = os.path.relpath(img_path, start='downloads')
                writer.writerow([rel_path, pose])
        
        print(f"Saved CSV file: {csv_path}")

def main():
    """Main function"""
    args = parse_args()
    
    # Parse pose classes
    pose_classes = [c.strip() for c in args.classes.split(',')]
    print(f"Processing the following pose classes: {pose_classes}")
    
    # Create downloads directory
    os.makedirs('downloads', exist_ok=True)
    
    # Clean if requested
    if args.clean:
        print("Cleaning existing downloaded images...")
        for pose in pose_classes:
            pose_dir = os.path.join('downloads', pose)
            if os.path.exists(pose_dir):
                shutil.rmtree(pose_dir)
    
    # Copy existing dataset
    print("Copying existing dataset...")
    existing_images = copy_existing_dataset(pose_classes)
    
    # Download new images
    all_images = {}
    for pose in pose_classes:
        # Initialize with existing images
        all_images[pose] = existing_images.get(pose, [])
        
        if args.download:
            print(f"\nDownloading images for {pose}...")
            downloaded = download_pose_images(pose, args.max_per_pose, args.download, args.max_threads)
            all_images[pose].extend(downloaded)
            
            print(f"Total images for {pose}: {len(all_images[pose])}")
    
    # Print statistics
    print("\nImage statistics:")
    for pose, images in all_images.items():
        print(f"  {pose}: {len(images)} images")
    
    # Parse Yoga-82 split if requested
    yoga82_split = None
    if args.use_yoga82_split:
        print("\nParsing Yoga-82 train/test split...")
        yoga82_split = parse_yoga82_split_files(pose_classes)
        print(f"Found {len(yoga82_split)} files in the Yoga-82 splits")
    
    # Split into train/test sets
    train_images, test_images = split_dataset(all_images, args.train_ratio)
    
    print(f"\nSplit statistics:")
    print(f"  Training set: {len(train_images)} images")
    print(f"  Testing set: {len(test_images)} images")
    
    # Save to folders
    save_splits_to_folders(train_images, test_images)
    
    # Create class names JSON file
    class_names = sorted(pose_classes)
    with open('yoga_class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Skip keypoint detection if requested
    if args.no_keypoints:
        print("\nSkipping keypoint detection as requested")
        print("\nDataset preparation complete!")
        print("For training, run: python training.py")
        return
    
    # Process keypoints with MoveNet Thunder
    print("\nProcessing pose keypoints with MoveNet Thunder...")
    try:
        # Load the model
        interpreter = load_movenet_model()
        print("MoveNet Thunder model loaded successfully")
        
        # Process training images
        print(f"Processing {len(train_images)} training images")
        train_detections = []
        
        # Get class mapping for pose classes
        class_mapping = {name: idx for idx, name in enumerate(class_names)}
        
        with tqdm(total=len(train_images), desc="Processing train images") as pbar:
            for img_path, pose in train_images:
                class_id = class_mapping.get(pose, 0)
                detection = detect_pose(interpreter, img_path, pose, class_id)
                if detection:
                    train_detections.append(detection)
                pbar.update(1)
                
        # Process testing images
        print(f"Processing {len(test_images)} testing images")
        test_detections = []
        
        with tqdm(total=len(test_images), desc="Processing test images") as pbar:
            for img_path, pose in test_images:
                class_id = class_mapping.get(pose, 0)
                detection = detect_pose(interpreter, img_path, pose, class_id)
                if detection:
                    test_detections.append(detection)
                pbar.update(1)
        
        # Save keypoints to CSV files
        print("\nSaving keypoint data to CSV files...")
        write_keypoints_to_csv(train_detections, 'train_data.csv')
        write_keypoints_to_csv(test_detections, 'test_data.csv')
        
    except Exception as e:
        print(f"Error processing keypoints: {e}")
        print("Continuing with dataset preparation...")
    
    print("\nDataset preparation complete!")
    print("For training, run: python training.py")

if __name__ == '__main__':
    main()
