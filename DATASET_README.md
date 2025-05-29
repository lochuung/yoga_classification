# Yoga Pose Dataset Preparation Script

This script downloads yoga pose images from the Yoga-82 dataset and organizes them into a structured train/test dataset for training the yoga pose classification model.

## Features

- Downloads images from URLs in the Yoga-82 dataset
- Creates a structured directory for images in `yoga_poses/train` and `yoga_poses/test`
- Preserves existing train/test split for existing images
- Creates CSV files in `csv_per_pose` for each pose class
- Supports multiple pose classes

## Prerequisites

Install the required Python packages:

```bash
pip install requests pillow opencv-python tqdm pandas
```

## Usage

```bash
python get_dataset.py [--classes CLASSES] [--download] [--train-ratio TRAIN_RATIO] [--clean] [--max-per-pose MAX_PER_POSE]
```

### Arguments:

- `--classes` - Comma-separated list of pose names to download (default: chair,cobra,dog,warrior,tree,traingle,shoudler_stand,no_pose)
- `--download` - Force download from URLs even if images exist locally
- `--train-ratio` - Ratio of images to use for training (default: 0.8)
- `--clean` - Remove existing downloaded images before downloading
- `--max-per-pose` - Maximum number of images to download per pose class (default: 500)

### Examples:

1. Download and prepare the full dataset:
   ```bash
   python get_dataset.py --download
   ```

2. Download only specific pose classes:
   ```bash
   python get_dataset.py --classes chair,cobra,tree --download
   ```

3. Remove all downloaded images and start fresh:
   ```bash
   python get_dataset.py --clean --download
   ```

4. Use a different train/test split ratio:
   ```bash
   python get_dataset.py --train-ratio 0.7 --download
   ```

## Output Structure

After running the script, the following directories will be created/updated:

- `downloads/` - Raw downloaded images
- `yoga_poses/train/` - Training dataset organized by pose
- `yoga_poses/test/` - Testing dataset organized by pose
- `csv_per_pose/` - CSV files for each pose with image paths

## Dataset Sources

This script uses pose data from:

- [Yoga-82: A New Dataset for Fine-grained Classification of Human Poses](https://arxiv.org/abs/2004.10362)

Please cite the original dataset paper if you use this data for research purposes:

```
@article{verma2020yoga,
  title={Yoga-82: A New Dataset for Fine-grained Classification of Human Poses},
  author={Verma, Manisha and Kumawat, Sudhakar and Nakashima, Yuta and Raman, Shanmuganathan},
  journal={arXiv preprint arXiv:2004.10362},
  year={2020}
}
```
