# Android Integration Guide for Yoga Pose Classifier

This document provides instructions for integrating the yoga pose classification model into your Android application.

## Prerequisites

- Android Studio 4.1 or higher
- Minimum SDK version 21 (Android 5.0)
- Basic knowledge of Android development

## Integration Steps

### 1. Add TensorFlow Lite Dependencies

Add the following to your app's `build.gradle` file:

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.4.0'
    
    // Optional: GPU acceleration support
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.4.0'
}
```

### 2. Copy Model Files to Assets

Copy the following files to your app's `assets` folder:
- `yoga_model.tflite` (regular model)
- `yoga_model_quantized.tflite` (optimized for slower devices)
- `yoga_class_names.json`
- `yoga_model_metadata.json`

### 3. Load and Use the Model

#### Java Implementation

```java
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import org.json.JSONArray;

public class YogaPoseClassifier {
    private Interpreter tflite;
    private String[] classNames;
    private static final int INPUT_SIZE = 34; // 17 keypoints x 2 coordinates
    
    // Load model from assets
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    // Load class names from JSON
    private String[] loadClassNames(Context context) throws IOException, JSONException {
        InputStream is = context.getAssets().open("yoga_class_names.json");
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            sb.append(line);
        }
        reader.close();
        
        JSONArray jsonArray = new JSONArray(sb.toString());
        String[] classNames = new String[jsonArray.length()];
        for (int i = 0; i < jsonArray.length(); i++) {
            classNames[i] = jsonArray.getString(i);
        }
        return classNames;
    }
    
    // Constructor
    public YogaPoseClassifier(Context context, boolean useGPU) throws IOException, JSONException {
        // Load class names
        classNames = loadClassNames(context);
        
        // Create interpreter options
        Interpreter.Options options = new Interpreter.Options();
        if (useGPU) {
            // Configure GPU acceleration if available
            GpuDelegate gpuDelegate = new GpuDelegate();
            options.addDelegate(gpuDelegate);
        }
        
        // Create the interpreter
        tflite = new Interpreter(loadModelFile(context, "yoga_model.tflite"), options);
    }
    
    // Classify a pose from normalized landmarks
    public YogaPoseResult classify(float[] normalizedLandmarks) {
        if (normalizedLandmarks.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Input must be 34 values (17 keypoints x 2 coordinates)");
        }
        
        // Prepare input and output buffers
        float[][] inputArray = new float[1][INPUT_SIZE];
        float[][] outputArray = new float[1][classNames.length];
        
        // Copy data to input array
        System.arraycopy(normalizedLandmarks, 0, inputArray[0], 0, INPUT_SIZE);
        
        // Run inference
        tflite.run(inputArray, outputArray);
        
        // Find the class with highest confidence
        int maxIndex = 0;
        float maxConfidence = outputArray[0][0];
        for (int i = 1; i < classNames.length; i++) {
            if (outputArray[0][i] > maxConfidence) {
                maxConfidence = outputArray[0][i];
                maxIndex = i;
            }
        }
        
        return new YogaPoseResult(classNames[maxIndex], maxConfidence);
    }
    
    // Result class
    public static class YogaPoseResult {
        public final String poseName;
        public final float confidence;
        
        public YogaPoseResult(String poseName, float confidence) {
            this.poseName = poseName;
            this.confidence = confidence;
        }
    }
    
    // Clean up resources
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }
}
```

#### Kotlin Implementation

```kotlin
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.content.Context
import android.content.res.AssetFileDescriptor
import org.json.JSONArray
import org.tensorflow.lite.gpu.GpuDelegate

class YogaPoseClassifier(
    context: Context,
    useGPU: Boolean = false
) {
    private val interpreter: Interpreter
    private val classNames: Array<String>
    private val inputSize = 34 // 17 keypoints x 2 coordinates
    
    init {
        // Load class names from JSON
        val classNamesJson = context.assets.open("yoga_class_names.json").bufferedReader().use { it.readText() }
        val jsonArray = JSONArray(classNamesJson)
        classNames = Array(jsonArray.length()) { i -> jsonArray.getString(i) }
        
        // Setup interpreter options
        val options = Interpreter.Options().apply {
            if (useGPU) {
                // Use GPU acceleration if requested
                addDelegate(GpuDelegate())
            }
            // Use 2 threads for inference
            numThreads = 2
        }
        
        // Create interpreter with model
        val modelBuffer = loadModelFile(context, "yoga_model.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }
    
    // Load TFLite model
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    // Classify pose from normalized landmarks
    fun classify(normalizedLandmarks: FloatArray): YogaPoseResult {
        require(normalizedLandmarks.size == inputSize) { "Input must be 34 values (17 keypoints x 2 coordinates)" }
        
        val inputArray = Array(1) { normalizedLandmarks }
        val outputArray = Array(1) { FloatArray(classNames.size) }
        
        // Run inference
        interpreter.run(inputArray, outputArray)
        
        // Find highest confidence class
        val results = outputArray[0]
        val maxConfidenceIndex = results.indices.maxByOrNull { results[it] } ?: 0
        
        return YogaPoseResult(
            poseName = classNames[maxConfidenceIndex],
            confidence = results[maxConfidenceIndex]
        )
    }
    
    // Result data class
    data class YogaPoseResult(
        val poseName: String,
        val confidence: Float
    )
    
    // Clean up resources
    fun close() {
        interpreter.close()
    }
}
```

### 4. Process Model Input

To use the model, you need to normalize pose landmarks from the pose detection model (like MoveNet):

```kotlin
// Example of processing landmarks from MoveNet
fun processLandmarks(landmarks: List<Pair<Float, Float>>): FloatArray {
    // Normalize landmarks following the same logic used in training
    val normalizedLandmarks = normalizeLandmarks(landmarks)
    
    // Convert to flat array of coordinates
    val flattenedLandmarks = FloatArray(normalizedLandmarks.size * 2)
    normalizedLandmarks.forEachIndexed { index, point ->
        flattenedLandmarks[index * 2] = point.first
        flattenedLandmarks[index * 2 + 1] = point.second
    }
    
    return flattenedLandmarks
}
```

### 5. Example Usage

Here's a complete example of how to use the classifier:

```kotlin
// Setup classifier when activity is created
private lateinit var yogaClassifier: YogaPoseClassifier

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // ...
    
    // Initialize classifier
    yogaClassifier = YogaPoseClassifier(this, useGPU = true)
}

// Process pose landmarks and classify yoga pose
fun classifyPose(landmarks: List<Pair<Float, Float>>) {
    // Process landmarks to get normalized coordinates
    val normalizedLandmarks = processLandmarks(landmarks)
    
    // Classify the pose
    val result = yogaClassifier.classify(normalizedLandmarks)
    
    // Show results
    if (result.confidence > 0.7) {
        poseNameTextView.text = result.poseName
        confidenceTextView.text = "Confidence: ${(result.confidence * 100).toInt()}%"
    } else {
        poseNameTextView.text = "No pose detected"
        confidenceTextView.text = ""
    }
}

// Clean up resources
override fun onDestroy() {
    yogaClassifier.close()
    super.onDestroy()
}
```

## Troubleshooting

### Common Issues

1. **Out of memory error**: Try using the quantized model `yoga_model_quantized.tflite` instead.

2. **Slow inference**: Enable GPU acceleration or increase the number of threads.

3. **Poor accuracy**: Make sure you're normalizing landmarks correctly.

4. **TFLite errors**: Ensure your TensorFlow Lite version matches the one used for training.

## Performance Tips

1. **Use GPU Delegate** for faster inference on compatible devices.

2. **Use the quantized model** on older or less powerful devices.

3. **Adjust inference thread count** based on your app's needs and device capabilities.

4. **Process frames at an appropriate rate** - you don't need to classify every camera frame.
