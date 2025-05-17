@echo off
REM Setup conda environment for yoga model training
echo Setting up conda environment for yoga pose classification model training...

REM Create a new conda environment with Python 3.8 (compatible with TensorFlow 2.4.0)
call conda create -y -n yoga-model python=3.8

REM Activate the environment 
call conda activate yoga-model

REM Install TensorFlow and other dependencies
call pip install -r requirements.txt

echo.
echo Environment setup complete! You can now run:
echo conda activate yoga-model
echo python training.py
echo.
echo After training, the following files will be generated:
echo - yoga_model.tflite (TensorFlow Lite model for Android)
echo - yoga_model_quantized.tflite (Optimized model for slower devices)
echo - yoga_class_names.json (Class names mapping)
echo - yoga_model_metadata.json (Model metadata)
echo.
echo Press any key to exit...
pause > nul
