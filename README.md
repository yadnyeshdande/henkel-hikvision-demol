# YOLO Trainer Tester — class mapping & verification

This project expects the YOLO model to use the following class mapping:
- class 0 = blue_box
- class 1 = yellow_mark

Files added by the assistant:
- `data.yaml` — minimal dataset config for training (train/val paths, nc, names).
- `verify_model.py` — small script to load a model and print `model.names` and any detections on an image.

Quick usage:
1. Ensure your training labels use YOLO format (one .txt per image):
   class x_center y_center width height  (all normalized [0..1])
   - Use `0` for blue_box labels and `1` for yellow_mark labels.

2. Train your model using Ultralytics (example):
   # adjust command depending on your setup and ultralytics version

3. After training, verify class names and a sample inference:

```powershell
python verify_model.py runs\train\weights\best.pt reference_frame.jpg
```

Notes:
- The app uses a confidence threshold of 0.5 in `henkel_2.py` when filtering detections. If you change that in the app, update verification accordingly.
- `data.yaml` uses relative paths; update them if your dataset is located elsewhere.
