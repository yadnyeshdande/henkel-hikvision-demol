"""
Small verification script to confirm class-id → class-name mapping in a trained ultralytics YOLO model
Usage:
  python verify_model.py [model_path] [image_path]
If omitted, defaults to `yolo11n.pt` and `reference_frame.jpg`.

This prints model.names and any detections (class id, class name, confidence, bbox).
"""
import sys
from ultralytics import YOLO


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'yolo11n.pt'
    image_path = sys.argv[2] if len(sys.argv) > 2 else 'reference_frame.jpg'

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Try several places where ultralytics stores class names
    names = None
    try:
        # v8+ models sometimes expose model.model.names
        names = getattr(model, 'names', None)
        if not names and hasattr(model, 'model'):
            names = getattr(model.model, 'names', None)
    except Exception:
        names = None

    print("Model class names (index:name):")
    if isinstance(names, dict):
        for k, v in names.items():
            print(f"  {k}: {v}")
    elif isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            print(f"  {i}: {v}")
    else:
        print("  (no names found on model object)")

    print('\nRunning a quick inference on:', image_path)
    try:
        results = model(image_path, verbose=False)
        for res in results:
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                continue
            for b in boxes:
                try:
                    cls = int(b.cls[0])
                except Exception:
                    cls = int(b.cls)
                # confidence attribute name can vary
                conf = None
                if hasattr(b, 'conf'):
                    conf = float(b.conf[0]) if hasattr(b.conf, '__len__') else float(b.conf)
                elif hasattr(b, 'confidence'):
                    conf = float(b.confidence[0]) if hasattr(b.confidence, '__len__') else float(b.confidence)

                xyxy = None
                try:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                except Exception:
                    try:
                        xyxy = b.xyxy.tolist()
                    except Exception:
                        xyxy = None

                name = None
                if names is not None:
                    try:
                        name = names[cls]
                    except Exception:
                        name = str(cls)
                else:
                    name = str(cls)

                print(f"Detected -> class_id={cls}, name={name}, conf={conf}, bbox={xyxy}")
    except Exception as e:
        print('Inference error:', e)


if __name__ == '__main__':
    main()
