# Setup Guide for Miniconda Users

This guide will help you set up a conda environment to train the yoga pose classification model.

## Prerequisites
- Miniconda or Anaconda installed on your system
- Git (to clone this repository, if you haven't already)

## Setup Steps

### Windows Setup

1. Open Command Prompt or PowerShell
2. Navigate to the project directory:
   ```
   cd path\to\yoga_model
   ```
3. Run the setup script:
   ```
   setup_conda_env.bat
   ```
4. After setup is complete, activate the environment:
   ```
   conda activate yoga-model
   ```
5. Run the training script:
   ```
   python training.py
   ```

### Mac/Linux Setup

1. Open Terminal
2. Navigate to the project directory:
   ```
   cd path/to/yoga_model
   ```
3. Make the setup script executable:
   ```
   chmod +x setup_conda_env.sh
   ```
4. Run the setup script:
   ```
   ./setup_conda_env.sh
   ```
5. After setup is complete, activate the environment:
   ```
   conda activate yoga-model
   ```
6. Run the training script:
   ```
   python training.py
   ```

## Manual Setup

If you prefer to set up manually, follow these steps:

1. Create a new conda environment:
   ```
   conda create -n yoga-model python=3.9
   ```
2. Activate the environment:
   ```
   conda activate yoga-model
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```
   python training.py
   ```

## Troubleshooting

If you encounter issues with TensorFlow installation:

1. Try installing TensorFlow via conda instead of pip:
   ```
   conda install tensorflow=2.10.0
   ```

2. If GPU support is needed, install the GPU version:
   ```
   conda install tensorflow-gpu=2.10.0
   ```

3. For specific versions of CUDA and cuDNN required by TensorFlow, refer to:
   https://www.tensorflow.org/install/source#gpu

4. If you're still having issues, try a slightly older version of TensorFlow:
   ```
   pip install tensorflow==2.8.0
   ```

5. Update your conda before installing packages:
   ```
   conda update -n base -c defaults conda
   ```
