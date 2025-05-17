"""
Yoga Pose Classification Model Training Script
--------------------------------------------
This script trains a model for yoga pose classification and exports it in TensorFlow Lite 
format for mobile deployment.

Required libraries:
- tensorflow==2.4.0 (pip install tensorflow==2.4.0)
- pandas==1.2.4 (pip install pandas==1.2.4)
- scikit-learn==0.24.2 (pip install scikit-learn==0.24.2)
- numpy==1.19.5 (pip install numpy==1.19.5)

Note: For compatibility with TensorFlow 2.4.0, use Python 3.6-3.8

Usage:
1. Ensure all dependencies are installed:
   pip install -r requirements.txt
   
   Or use conda:
   conda create -n yoga-model python=3.8
   conda activate yoga-model
   pip install -r requirements.txt

2. Run the script:
   python training.py

3. After successful training, the following files will be generated:
   - yoga_model.tflite (TensorFlow Lite model for Android)
   - yoga_model_quantized.tflite (Quantized TFLite model for better performance)
   - yoga_class_names.json (Class names mapping)
   - yoga_model_metadata.json (Model metadata for Android integration)

4. Copy the .tflite and .json files to your Android project's assets folder.
   See ANDROID_INTEGRATION.md for detailed integration instructions.
"""

import csv
import os
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from data import BodyPart 
import tensorflow as tf
import json

# Set model paths
tflite_model_file = 'yoga_model.tflite'  # File to save the TFLite model


# loading final csv file
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(['filename'],axis=1, inplace=True)
    classes = df.pop('class_name').unique()
    y = df.pop('class_no')
    
    X = df.astype('float64')
    y = keras.utils.to_categorical(y)
    
    return X, y, classes


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                     BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size



def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)

    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    return embedding


def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)


X, y, class_names = load_csv('train_data.csv')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
X_test, y_test, _ = load_csv('test_data.csv')


processed_X_train = preprocess_data(X_train)
processed_X_val =  preprocess_data(X_val)
processed_X_test = preprocess_data(X_test)

inputs = tf.keras.Input(shape=(34))
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(inputs)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint_path = "weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                              patience=20)

# Start training
print('--------------TRAINING----------------')
history = model.fit(processed_X_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(processed_X_val, y_val),
                    callbacks=[checkpoint, earlystopping])


print('-----------------EVAUATION----------------')
loss, accuracy = model.evaluate(processed_X_test, y_test)
print('LOSS: ', loss)
print("ACCURACY: ", accuracy)

# Convert the model to TFLite format
print('-----------------TFLITE CONVERSION----------------')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization options for better performance on mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Apply quantization to reduce model size and improve inference speed
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 
                                      tf.lite.OpsSet.SELECT_TF_OPS]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)
print(f'TFLite model for Android saved at {tflite_model_file}')

# Create a quantized model for better performance on mobile
print('-----------------CREATING OPTIMIZED MODEL----------------')
try:
    # Try to create a quantized model
    quantized_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quantized_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization
    def representative_dataset():
        processed_data = processed_X_val.numpy()
        for i in range(min(100, len(processed_data))):
            yield [processed_data[i:i+1].astype('float32')]
    
    quantized_converter.representative_dataset = representative_dataset
    quantized_model = quantized_converter.convert()
    
    # Save the quantized model
    quantized_model_file = 'yoga_model_quantized.tflite'
    with open(quantized_model_file, 'wb') as f:
        f.write(quantized_model)
    print(f'Optimized TFLite model saved at {quantized_model_file}')
except Exception as e:
    print(f"Quantization skipped due to error: {e}")
    quantized_model_file = None

# Export the class names for use in Android
class_names_file = 'yoga_class_names.json'
with open(class_names_file, 'w') as f:
    json.dump(class_names.tolist(), f)
print(f'Class names saved at {class_names_file}')

# Create a metadata file with model information for Android integration
metadata = {
    'model_type': 'yoga_pose_classifier',
    'input_shape': model.input_shape[1:],
    'output_shape': model.output_shape[1:],
    'num_classes': len(class_names),
    'classes': class_names.tolist(),
    'input_tensor_name': 'serving_default_input_1:0',
    'output_tensor_name': 'StatefulPartitionedCall:0',
    'preprocessing': 'normalized_landmarks',
    'model_versions': {
        'standard': tflite_model_file,
        'quantized': quantized_model_file
    },
    'android_info': {
        'min_sdk_version': 21,
        'target_sdk_version': 30,
        'recommended_inference_accelerator': 'GPU'
    }
}

metadata_file = 'yoga_model_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'Model metadata saved at {metadata_file}')

# Print model details for reference
print(f'Number of yoga poses: {len(class_names)}')
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')

print('\nCOMPLETE: Your yoga model is now ready for Android Studio!')
print('Copy the following files to your Android project\'s assets folder:')
print(f'- {tflite_model_file}')
print(f'- {quantized_model_file}')
print(f'- {class_names_file}')
print(f'- {metadata_file}')

# Add instructions for Android Studio integration
print('\nTo use the model in Android Studio:')
print('1. Add the TensorFlow Lite dependency in your app\'s build.gradle file:')
print('   implementation "org.tensorflow:tensorflow-lite:2.4.0"')
print('2. Load the model in your Android code:')
print('   val interpreter = Interpreter(loadModelFile(context, "yoga_model.tflite"))')
print('3. Process input data (normalized landmarks) and run inference')
print('4. Parse the output to get the predicted yoga pose')
print('5. Refer to the metadata.json file for input/output tensor details')
