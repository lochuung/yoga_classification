@echo off
echo Testing yoga model with fixed keypoint orientation...
python test_yoga_model.py --use_quantized --source image --path "yoga_poses/test/tree/guy3_tree085.jpg"
pause
