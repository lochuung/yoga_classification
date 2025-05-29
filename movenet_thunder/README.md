# Yoga Pose Classification Model

This project trains a machine learning model for yoga pose classification and exports it for use in Android applications.

## Project Structure

- `training.py`: Main script for training the yoga pose classification model
- `data.py`: Helper module with data types for pose estimation
- `train_data.csv` & `test_data.csv`: Dataset for training and testing
- `requirements.txt`: Required Python dependencies

## Setup and Training

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python training.py
```

After training completes, the following files will be generated:
- `yoga_model.tflite` - TensorFlow Lite model for Android
- `yoga_model_quantized.tflite` - Quantized TensorFlow Lite model for slower devices
- `yoga_class_names.json` - Class names mapping
- `yoga_model_metadata.json` - Model metadata

## Android Integration

See [ANDROID_INTEGRATION.md](ANDROID_INTEGRATION.md) for detailed instructions on how to integrate the model into your Android application.

### Quick Start (Android)

1. Add TensorFlow Lite to your app's build.gradle dependencies:
```gradle
implementation 'org.tensorflow:tensorflow-lite:2.4.0'
```

## Using the model in Android Studio

1. Create an Android project in Android Studio

2. Copy the generated files to your Android project's assets folder:
   - `yoga_model.tflite`
   - `yoga_class_names.json`
   - `yoga_model_metadata.json`

3. Add TensorFlow Lite to your app's build.gradle dependencies:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.8.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.0'
}
```

4. Load and use the model in your Android code:

```java
// Example loading and using the model in Java
try {
    // Load the TFLite model
    Interpreter tflite = new Interpreter(loadModelFile(activity, "yoga_model.tflite"));
    
    // Process input data (normalized pose landmarks)
    float[][] input = new float[1][34]; // Shape matches model input
    
    // Run inference
    float[][] output = new float[1][NUM_CLASSES];
    tflite.run(input, output);
    
    // Process results
    int maxIndex = 0;
    float maxValue = output[0][0];
    for (int i = 1; i < output[0].length; i++) {
        if (output[0][i] > maxValue) {
            maxValue = output[0][i];
            maxIndex = i;
        }
    }
    
    // maxIndex now contains the predicted pose class
    String predictedPose = classNames[maxIndex];
    
} catch (IOException e) {
    Log.e("YogaApp", "Error loading model", e);
}
```

```kotlin
// Example loading and using the model in Kotlin
try {
    // Load the TFLite model
    val tflite = Interpreter(loadModelFile(activity, "yoga_model.tflite"))
    
    // Process input data (normalized pose landmarks)
    val input = Array(1) { FloatArray(34) } // Shape matches model input
    
    // Run inference
    val output = Array(1) { FloatArray(NUM_CLASSES) }
    tflite.run(input, output)
    
    // Process results - find index with highest confidence
    val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: 0
    
    // maxIndex now contains the predicted pose class
    val predictedPose = classNames[maxIndex]
    
} catch (e: IOException) {
    Log.e("YogaApp", "Error loading model", e)
}
```

## License

This project uses code from the TensorFlow example repository, which is licensed under the Apache License 2.0.
