"""
Yoga Pose Classification Model Testing Script
--------------------------------------------
This script tests the yoga pose classification model using either:
1. Camera input (webcam)
2. Image files from test directory

Requirements:
- tensorflow==2.4.0
- opencv-python==4.5.3.56
- numpy==1.19.5

Usage:
1. Test using webcam:
   python test_yoga_model.py --source webcam

2. Test using a specific image:
   python test_yoga_model.py --source image --path "yoga_poses/test/tree/image001.jpg"

3. Test using all images in a test folder:
   python test_yoga_model.py --source folder --path "yoga_poses/test/tree"
"""

import os
import argparse
import json
import numpy as np
import cv2
import tensorflow as tf
from data import BodyPart

# Constants
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 0),
    (0, 2): (0, 0, 255),
    (1, 3): (255, 0, 0),
    (2, 4): (0, 0, 255),
    (0, 5): (255, 0, 0),
    (0, 6): (0, 0, 255),
    (5, 7): (255, 0, 0),
    (7, 9): (255, 0, 0),
    (6, 8): (0, 0, 255),
    (8, 10): (0, 0, 255),
    (5, 6): (0, 255, 0),
    (5, 11): (255, 0, 0),
    (6, 12): (0, 0, 255),
    (11, 12): (0, 255, 0),
    (11, 13): (255, 0, 0),
    (13, 15): (255, 0, 0),
    (12, 14): (0, 0, 255),
    (14, 16): (0, 0, 255)
}


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test Yoga Pose Classification Model')
    parser.add_argument('--source', type=str, default='webcam',
                        choices=['webcam', 'image', 'folder'],
                        help='Source for testing: webcam, image, or folder')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to image or folder for testing')
    parser.add_argument('--model', type=str, default='yoga_model.tflite',
                        help='Path to yoga classification model. Try yoga_model_quantized.tflite for quantized model.')
    parser.add_argument('--pose_model', type=str, default='movenet_thunder.tflite',
                        help='Path to pose detection model')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Confidence threshold for pose detection')
    parser.add_argument('--use_quantized', action='store_true',
                        help='Use the quantized model (yoga_model_quantized.tflite) for better compatibility')
    return parser.parse_args()


# Load TFLite model and allocate tensors
def load_pose_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# Load the yoga pose classification model
def load_yoga_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print model details for debugging
    print(f"Model input details: {input_details}")
    print(f"Model input shape: {input_details[0]['shape']}")
    print(f"Model input type: {input_details[0]['dtype']}")
    if 'quantization' in input_details[0]:
        print(f"Model quantization parameters: {input_details[0]['quantization']}")

    return interpreter, input_details, output_details


# Load class names
def load_class_names(json_path='yoga_class_names.json'):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names


def rotate_keypoints_90_clockwise(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    for i, (x, y) in enumerate(keypoints):
        new_keypoints[i][0] = y
        new_keypoints[i][1] = 1 - x
    return new_keypoints


# Function to run pose detection
def detect_pose(interpreter, input_image):
    """Run pose detection on input image and return keypoints."""
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input image
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]

    img = cv2.resize(input_image, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Check for required data type
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
        # Quantize the input for uint8 models (like MoveNet)
        img = img.astype(np.float32)
        input_data = np.expand_dims(img, axis=0)

        # Check if quantization parameters are available
        if 'quantization' in input_details[0] and input_details[0]['quantization'][0] != 0:
            scale, zero_point = input_details[0]['quantization']
        else:
            # Default quantization for 0-255 range if not specified
            scale, zero_point = 1 / 255.0, 0

        # Apply quantization
        input_data = input_data / scale + zero_point
        input_data = input_data.astype(np.uint8)
    else:
        # For float models
        img = img.astype(np.float32) / 255.0
        input_data = np.expand_dims(img, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


# Function to normalize landmarks
def normalize_landmarks(landmarks):
    """Normalize the landmarks using the same logic as in training."""
    # Get hip center
    left_hip = landmarks[BodyPart.LEFT_HIP.value][:2]
    right_hip = landmarks[BodyPart.RIGHT_HIP.value][:2]
    hip_center = (left_hip + right_hip) / 2

    # Get shoulder center
    left_shoulder = landmarks[BodyPart.LEFT_SHOULDER.value][:2]
    right_shoulder = landmarks[BodyPart.RIGHT_SHOULDER.value][:2]
    shoulder_center = (left_shoulder + right_shoulder) / 2

    # Calculate torso size
    torso_size = np.linalg.norm(shoulder_center - hip_center)

    # Center all landmarks around hip center
    centered_landmarks = landmarks[:, :2] - hip_center

    # Find max distance from center
    max_dist = np.max(np.linalg.norm(centered_landmarks, axis=1))

    # Use the larger of torso size and max distance for normalization
    pose_size = max(torso_size * 2.5, max_dist)

    # Normalize
    normalized_landmarks = centered_landmarks / pose_size if pose_size > 0 else centered_landmarks

    return normalized_landmarks.flatten()


# Function to draw pose landmarks on image
def draw_pose(image, keypoints_with_scores, threshold=0.25):
    """Draw pose landmarks and connections on the image."""
    height, width, _ = image.shape
    keypoints = keypoints_with_scores[0, 0, :, :2].copy()

    # Flip horizontally (mirror image)
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        keypoints[i][0] = 1 - y
        keypoints[i][1] = x
        keypoints[i][0] = 1 - keypoints[i][0]

    # Scale to image dimensions
    keypoints = keypoints * np.array([width, height])
    scores = keypoints_with_scores[0, 0, :, 2]

    # Draw the points
    for idx, (point, score) in enumerate(zip(keypoints, scores)):
        if score > threshold:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, f"{idx}", (x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw the connections
    for connection, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        start_idx, end_idx = connection
        if scores[start_idx] > threshold and scores[end_idx] > threshold:
            start_point = tuple(map(int, keypoints[start_idx]))
            end_point = tuple(map(int, keypoints[end_idx]))
            cv2.line(image, start_point, end_point, color, 2)

    return image



# Main test function for a single image
def test_on_image(image, pose_interpreter, yoga_interpreter, input_details, output_details, class_names,
                  threshold=0.25):
    # Detect pose
    keypoints_with_scores = detect_pose(pose_interpreter, image)

    # Draw pose
    result_image = image.copy()
    result_image = draw_pose(result_image, keypoints_with_scores, threshold)

    # Check if a pose is detected with sufficient confidence
    scores = keypoints_with_scores[0, 0, :, 2]
    if np.mean(scores) < threshold:
        cv2.putText(result_image, "No pose detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return result_image, None, None

    # Prepare landmarks for classification
    landmarks = keypoints_with_scores[0, 0, :, :]
    normalized_landmarks = normalize_landmarks(landmarks)
    # Reshape for model input
    input_tensor = np.expand_dims(normalized_landmarks, axis=0)

    # Check the required data type from the model
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
        # Quantize the input if model requires uint8
        # Assuming input range is [-1, 1] for normalized landmarks
        input_tensor = input_tensor.astype(np.float32)
        scale, zero_point = input_details[0]['quantization']
        if scale != 0:  # Ensure we don't divide by zero
            input_tensor = input_tensor / scale + zero_point
        input_tensor = input_tensor.astype(np.uint8)
    else:
        # Otherwise use float32
        input_tensor = input_tensor.astype(np.float32)

    # Set input tensor and run inference
    yoga_interpreter.set_tensor(input_details[0]['index'], input_tensor)
    yoga_interpreter.invoke()

    # Get results
    output = yoga_interpreter.get_tensor(output_details[0]['index'])
    predicted_idx = np.argmax(output[0])
    confidence = output[0][predicted_idx]
    predicted_pose = class_names[predicted_idx]

    # Draw prediction
    cv2.putText(result_image, f"Pose: {predicted_pose}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_image, f"Confidence: {confidence:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result_image, predicted_pose, confidence


# Test using webcam
def test_webcam(pose_interpreter, yoga_interpreter, input_details, output_details, class_names, threshold=0.25):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result_frame, pose, confidence = test_on_image(
            frame, pose_interpreter, yoga_interpreter, input_details, output_details, class_names, threshold)

        # Display result
        cv2.imshow('Yoga Pose Classification', result_frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Test using a specific image
def test_single_image(image_path, pose_interpreter, yoga_interpreter, input_details, output_details, class_names,
                      threshold=0.25):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = cv2.imread(image_path)
    # No need to rotate the image since we're fixing the keypoints in draw_pose
    result_image, pose, confidence = test_on_image(
        image, pose_interpreter, yoga_interpreter, input_details, output_details, class_names, threshold)

    # Display result
    cv2.imshow('Yoga Pose Classification', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if pose:
        print(f"Detected Pose: {pose} (Confidence: {confidence:.2f})")
    else:
        print("No pose detected with sufficient confidence")


# Test using images in a folder
def test_folder(folder_path, pose_interpreter, yoga_interpreter, input_details, output_details, class_names,
                threshold=0.25):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in image_extensions]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    results = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        # No need to rotate the image since we're fixing the keypoints in draw_pose
        result_image, pose, confidence = test_on_image(
            image, pose_interpreter, yoga_interpreter, input_details, output_details, class_names, threshold)

        # Display result
        cv2.imshow('Yoga Pose Classification', result_image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

        if pose:
            print(f"{image_file}: {pose} (Confidence: {confidence:.2f})")
            results.append((image_file, pose, confidence))
        else:
            print(f"{image_file}: No pose detected")

    cv2.destroyAllWindows()

    # Summary of results
    if results:
        print("\nSummary:")
        for image_file, pose, confidence in results:
            print(f"{image_file}: {pose} (Confidence: {confidence:.2f})")


# Main function
def main():
    args = parse_args()

    # Use quantized model if specified
    if args.use_quantized and args.model == 'yoga_model.tflite':
        args.model = 'yoga_model_quantized.tflite'
        print("Using quantized model: yoga_model_quantized.tflite")

    # Load pose detection model
    print("Loading pose detection model...")
    pose_interpreter = load_pose_model(args.pose_model)

    # Load yoga classification model
    print(f"Loading yoga classification model: {args.model}...")
    yoga_interpreter, input_details, output_details = load_yoga_model(args.model)

    # Load class names
    class_names = load_class_names()
    print(f"Loaded {len(class_names)} yoga poses: {', '.join(class_names)}")
    # Run appropriate test based on source
    if args.source == 'webcam':
        print("Starting webcam test. Press 'q' to quit.")
        test_webcam(pose_interpreter, yoga_interpreter, input_details, output_details, class_names, args.conf_threshold)
    elif args.source == 'image':
        if not args.path:
            print("Error: Must specify --path when using --source image")
        else:
            print(f"Testing image: {args.path}")
            test_single_image(args.path, pose_interpreter, yoga_interpreter, input_details, output_details, class_names,
                              args.conf_threshold)
    elif args.source == 'folder':
        if not args.path:
            print("Error: Must specify --path when using --source folder")
        else:
            print(f"Testing images in folder: {args.path}")
            test_folder(args.path, pose_interpreter, yoga_interpreter, input_details, output_details, class_names,
                        args.conf_threshold)


if __name__ == "__main__":
    main()
