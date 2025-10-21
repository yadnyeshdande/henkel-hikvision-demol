import sys
import os
import cv2
import json
import yaml
import shutil
import threading
import tempfile
import re
from datetime import datetime
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTextEdit, QComboBox, QProgressBar, QFileDialog,
                             QMessageBox, QSpinBox, QGroupBox, QGridLayout,
                             QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
import subprocess
import time
import random

class AnnotationWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid gray;")
        self.setScaledContents(False)  # Don't stretch content
        self.setAlignment(Qt.AlignCenter)  # Center the image
        self.drawing = False
        self.current_rect = None
        self.rectangles = []
        self.start_point = None
        self.class_name = "object"
        
        # Image transformation properties
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.image_scale_x = 1.0
        self.image_scale_y = 1.0
        self.original_width = 0
        self.original_height = 0
        
    def set_class_name(self, name):
        self.class_name = name
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Adjust mouse position to account for image offset
            adjusted_x = event.pos().x() - self.image_offset_x
            adjusted_y = event.pos().y() - self.image_offset_y
            
            # Check if click is within the actual image bounds
            if (0 <= adjusted_x <= self.pixmap().width() if self.pixmap() else False and
                0 <= adjusted_y <= self.pixmap().height() if self.pixmap() else False):
                self.drawing = True
                self.start_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.drawing and self.start_point:
            # Constrain rectangle to image bounds
            pixmap = self.pixmap()
            if pixmap:
                max_x = self.image_offset_x + pixmap.width()
                max_y = self.image_offset_y + pixmap.height()
                min_x = self.image_offset_x
                min_y = self.image_offset_y
                
                constrained_x = max(min_x, min(max_x, event.pos().x()))
                constrained_y = max(min_y, min(max_y, event.pos().y()))
                
                from PyQt5.QtCore import QPoint
                constrained_pos = QPoint(constrained_x, constrained_y)
                self.current_rect = QRect(self.start_point, constrained_pos).normalized()
                self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_rect and self.current_rect.width() > 10 and self.current_rect.height() > 10:
                self.rectangles.append({
                    'rect': self.current_rect,
                    'class': self.class_name
                })
            self.current_rect = None
            self.update()
            
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        # Draw confirmed rectangles
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for rect_data in self.rectangles:
            painter.drawRect(rect_data['rect'])
            painter.drawText(rect_data['rect'].topLeft(), rect_data['class'])
            
        # Draw current rectangle being drawn
        if self.current_rect:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)
            
    def clear_annotations(self):
        self.rectangles = []
        self.current_rect = None
        self.update()
        
    def get_yolo_annotations(self, img_width, img_height):
        annotations = []
        
        if not self.pixmap():
            return annotations
            
        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()
        
        for rect_data in self.rectangles:
            rect = rect_data['rect']
            class_name = rect_data['class']
            
            # Convert widget coordinates to image coordinates
            # Account for image offset and scaling
            x_center_widget = rect.center().x() - self.image_offset_x
            y_center_widget = rect.center().y() - self.image_offset_y
            width_widget = rect.width()
            height_widget = rect.height()
            
            # Convert to original image coordinates
            x_center_img = x_center_widget / self.image_scale_x
            y_center_img = y_center_widget / self.image_scale_y
            width_img = width_widget / self.image_scale_x
            height_img = height_widget / self.image_scale_y
            
            # Normalize to YOLO format (0-1)
            x_center_norm = x_center_img / self.original_width
            y_center_norm = y_center_img / self.original_height
            width_norm = width_img / self.original_width
            height_norm = height_img / self.original_height
            
            # Clamp values to [0, 1] range
            x_center_norm = max(0, min(1, x_center_norm))
            y_center_norm = max(0, min(1, y_center_norm))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))
            
            annotations.append({
                'class': class_name,
                'x_center': x_center_norm,
                'y_center': y_center_norm,
                'width': width_norm,
                'height': height_norm
            })
            
        return annotations

class CaptureThread(QThread):
    image_captured = pyqtSignal(np.ndarray, str)
    capture_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.capturing = False
        self.annotations = []
        self.class_names = []
        self.output_dir = ""
        self.capture_interval = 0.1  # 100ms between captures
        
    def set_parameters(self, annotations, class_names, output_dir):
        self.annotations = annotations
        self.class_names = class_names
        self.output_dir = output_dir
        
    def start_capture(self):
        self.capturing = True
        self.start()
        
    def stop_capture(self):
        self.capturing = False
        
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
            
        counter = 0
        while self.capturing:
            ret, frame = cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"img_{counter:06d}_{timestamp}.jpg"
                counter += 1
                
                self.image_captured.emit(frame, filename)
                
            time.sleep(self.capture_interval)
            
        cap.release()
        self.capture_finished.emit()

class TrainingThread(QThread):
    training_progress = pyqtSignal(str)
    training_finished = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self.model_name = ""
        self.data_yaml = ""
        self.epochs = 100
        
    def set_parameters(self, model_name, data_yaml, epochs):
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        
    def run(self):
        try:
            # Install ultralytics if not present
            self.training_progress.emit("Installing/updating ultralytics...")
            subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], 
                          capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            # Start training
            self.training_progress.emit(f"Starting training with {self.model_name}...")
            
            # Create a temporary Python script to avoid string escaping issues
            script_content = f'''
import os
import sys
from ultralytics import YOLO

# Set environment variable to handle encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    model = YOLO("{self.model_name}.pt")
    results = model.train(data=r"{self.data_yaml}", epochs={self.epochs}, imgsz=640, save=True, verbose=True)
    print(f"Training completed. Results saved to: {{results.save_dir}}")
except Exception as e:
    print(f"Training error: {{e}}")
    sys.exit(1)
'''
            
            # Write script to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(script_content)
                script_path = f.name
            
            try:
                # Run the script
                process = subprocess.Popen(
                    [sys.executable, script_path], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    encoding='utf-8', 
                    errors='ignore',
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
                )
                
                output_lines = []
                for line in process.stdout:
                    line = line.strip()
                    if line:  # Only emit non-empty lines
                        output_lines.append(line)
                        self.training_progress.emit(line)
                    
                process.wait()
                
                # Check if training was successful
                if process.returncode != 0:
                    self.training_finished.emit("Training failed. Check the log for details.", "")
                    return
                
                # Find results directory - improved parsing
                results_dir = ""
                for line in output_lines:
                    if "Results saved to" in line:
                        # Extract path from colored text
                        import re
                        # Remove ANSI color codes
                        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                        if "Results saved to" in clean_line:
                            results_dir = clean_line.split("Results saved to")[-1].strip()
                            # Remove any remaining formatting
                            results_dir = results_dir.replace('[1m', '').replace('[0m', '').strip()
                            break
                        
                if not results_dir:
                    # Try to find the latest run directory
                    runs_dir = Path("runs/detect")
                    if runs_dir.exists():
                        train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]
                        if train_dirs:
                            results_dir = str(max(train_dirs, key=lambda x: x.stat().st_mtime))
                
                # Convert to absolute path
                if results_dir and not os.path.isabs(results_dir):
                    results_dir = os.path.abspath(results_dir)
                
                self.training_finished.emit("Training completed successfully!", results_dir)
                
            finally:
                # Clean up temporary script
                try:
                    os.unlink(script_path)
                except:
                    pass
            
        except Exception as e:
            self.training_finished.emit(f"Training failed: {str(e)}", "")

class YOLOTrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Training Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.camera = None
        self.current_image = None
        self.project_dir = ""
        self.class_names = ["object"]
        self.current_class_index = 0
        self.preview_active = False
        
        # Initialize threads
        self.capture_thread = CaptureThread()
        self.training_thread = TrainingThread()
        
        # Timer for camera preview
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Camera and annotation
        left_panel = QVBoxLayout()
        
        # Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout(camera_group)
        
        # Preview controls
        preview_layout = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.start_preview_btn.clicked.connect(self.start_preview)
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.stop_preview_btn.clicked.connect(self.stop_preview)
        self.stop_preview_btn.setEnabled(False)
        preview_layout.addWidget(self.start_preview_btn)
        preview_layout.addWidget(self.stop_preview_btn)
        camera_layout.addLayout(preview_layout)
        
        self.capture_single_btn = QPushButton("Capture Single Image")
        self.capture_single_btn.clicked.connect(self.capture_single_image)
        camera_layout.addWidget(self.capture_single_btn)
        
        # Annotation widget
        self.annotation_widget = AnnotationWidget()
        self.annotation_widget.setFixedSize(640, 480)  # Fixed size to prevent stretching
        
        # Class selection
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Class:"))
        self.class_input = QLineEdit("object")
        self.class_input.textChanged.connect(self.update_class_name)
        class_layout.addWidget(self.class_input)
        
        self.clear_annotations_btn = QPushButton("Clear Annotations")
        self.clear_annotations_btn.clicked.connect(self.annotation_widget.clear_annotations)
        class_layout.addWidget(self.clear_annotations_btn)
        
        # Confirm annotations
        self.confirm_annotations_btn = QPushButton("Confirm Annotations")
        self.confirm_annotations_btn.clicked.connect(self.confirm_annotations)
        
        left_panel.addWidget(camera_group)
        left_panel.addWidget(self.annotation_widget)
        left_panel.addLayout(class_layout)
        left_panel.addWidget(self.confirm_annotations_btn)
        
        # Right panel - Controls and status
        right_panel = QVBoxLayout()
        
        # Project setup
        project_group = QGroupBox("Project Setup")
        project_layout = QVBoxLayout(project_group)
        
        project_dir_layout = QHBoxLayout()
        project_dir_layout.addWidget(QLabel("Project Directory:"))
        self.project_dir_input = QLineEdit()
        project_dir_layout.addWidget(self.project_dir_input)
        self.browse_dir_btn = QPushButton("Browse")
        self.browse_dir_btn.clicked.connect(self.browse_project_directory)
        project_dir_layout.addWidget(self.browse_dir_btn)
        project_layout.addLayout(project_dir_layout)
        
        # Burst capture controls
        burst_group = QGroupBox("Burst Capture")
        burst_layout = QVBoxLayout(burst_group)
        
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Capture Interval (ms):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(50, 5000)
        self.interval_spin.setValue(100)
        interval_layout.addWidget(self.interval_spin)
        burst_layout.addLayout(interval_layout)
        
        burst_btn_layout = QHBoxLayout()
        self.start_burst_btn = QPushButton("Start Burst Capture")
        self.start_burst_btn.clicked.connect(self.start_burst_capture)
        self.stop_burst_btn = QPushButton("Stop Burst Capture")
        self.stop_burst_btn.clicked.connect(self.stop_burst_capture)
        self.stop_burst_btn.setEnabled(False)
        burst_btn_layout.addWidget(self.start_burst_btn)
        burst_btn_layout.addWidget(self.stop_burst_btn)
        burst_layout.addLayout(burst_btn_layout)
        
        self.capture_count_label = QLabel("Images captured: 0")
        burst_layout.addWidget(self.capture_count_label)
        
        # Training controls
        training_group = QGroupBox("Training Setup")
        training_layout = QVBoxLayout(training_group)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("YOLO Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
                                  "yolov9n", "yolov9s", "yolov9m", "yolov9l", "yolov9x",
                                  "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"])
        model_layout.addWidget(self.model_combo)
        training_layout.addLayout(model_layout)
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        epochs_layout.addWidget(self.epochs_spin)
        training_layout.addLayout(epochs_layout)
        
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Train/Test Split (%):"))
        self.split_spin = QSpinBox()
        self.split_spin.setRange(50, 95)
        self.split_spin.setValue(80)
        split_layout.addWidget(self.split_spin)
        training_layout.addLayout(split_layout)
        
        self.prepare_data_btn = QPushButton("Prepare Training Data")
        self.prepare_data_btn.clicked.connect(self.prepare_training_data)
        
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self.start_training)
        self.start_training_btn.setEnabled(False)
        
        training_layout.addWidget(self.prepare_data_btn)
        training_layout.addWidget(self.start_training_btn)
        
        # Progress and status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_text)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.open_results_btn = QPushButton("Open Results Folder")
        self.open_results_btn.clicked.connect(self.open_results_folder)
        self.open_results_btn.setEnabled(False)
        
        results_layout.addWidget(self.open_results_btn)
        
        # Add all groups to right panel
        right_panel.addWidget(project_group)
        right_panel.addWidget(burst_group)
        right_panel.addWidget(training_group)
        right_panel.addWidget(status_group)
        right_panel.addWidget(results_group)
        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        # Initialize camera
        self.initialize_camera()
        
    def connect_signals(self):
        self.capture_thread.image_captured.connect(self.on_image_captured)
        self.capture_thread.capture_finished.connect(self.on_capture_finished)
        
        self.training_thread.training_progress.connect(self.on_training_progress)
        self.training_thread.training_finished.connect(self.on_training_finished)
        
    def initialize_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.log_status("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.log_status("Camera initialized successfully")
        
    def start_preview(self):
        if not self.camera or not self.camera.isOpened():
            self.initialize_camera()
            
        if self.camera and self.camera.isOpened():
            self.preview_active = True
            self.preview_timer.start(33)  # ~30 FPS (33ms intervals)
            self.start_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(True)
            self.log_status("Camera preview started")
        else:
            self.log_status("Error: Could not start camera preview")
            
    def stop_preview(self):
        self.preview_active = False
        self.preview_timer.stop()
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.log_status("Camera preview stopped")
        
    def update_preview(self):
        if not self.camera or not self.camera.isOpened() or not self.preview_active:
            return
            
        ret, frame = self.camera.read()
        if ret:
            self.display_image(frame)
        else:
            self.log_status("Error: Failed to read camera frame")
            self.stop_preview()
        
    def capture_single_image(self):
        if not self.camera or not self.camera.isOpened():
            self.log_status("Error: Camera not available")
            return
            
        ret, frame = self.camera.read()
        if ret:
            self.current_image = frame
            # Stop preview temporarily to allow annotation
            if self.preview_active:
                self.preview_timer.stop()
            self.display_image(frame)
            self.log_status("Single image captured - Preview paused for annotation")
        else:
            self.log_status("Error: Failed to capture image")
            
    def display_image(self, image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        
        from PyQt5.QtGui import QImage
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale pixmap to fit the annotation widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.annotation_widget.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.annotation_widget.setPixmap(scaled_pixmap)
        
        # Store the scaling factors for accurate annotation conversion
        widget_width = self.annotation_widget.width()
        widget_height = self.annotation_widget.height()
        pixmap_width = scaled_pixmap.width()
        pixmap_height = scaled_pixmap.height()
        
        # Calculate offset for centering
        self.annotation_widget.image_offset_x = (widget_width - pixmap_width) // 2
        self.annotation_widget.image_offset_y = (widget_height - pixmap_height) // 2
        self.annotation_widget.image_scale_x = pixmap_width / width
        self.annotation_widget.image_scale_y = pixmap_height / height
        self.annotation_widget.original_width = width
        self.annotation_widget.original_height = height
        
    def update_class_name(self):
        class_name = self.class_input.text().strip()
        if class_name:
            self.annotation_widget.set_class_name(class_name)
            
    def confirm_annotations(self):
        if self.current_image is None:
            self.log_status("Error: No image captured")
            return
            
        annotations = self.annotation_widget.get_yolo_annotations(
            self.current_image.shape[1], self.current_image.shape[0])
        
        if not annotations:
            self.log_status("Error: No annotations found")
            return
            
        # Store annotations for burst capture
        self.confirmed_annotations = annotations
        self.log_status(f"Confirmed {len(annotations)} annotations")
        
        # Extract class names
        self.class_names = list(set([ann['class'] for ann in annotations]))
        self.log_status(f"Classes: {', '.join(self.class_names)}")
        
        # Resume preview if it was active
        if not self.preview_timer.isActive() and self.stop_preview_btn.isEnabled():
            self.preview_timer.start(33)
        
    def browse_project_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.project_dir_input.setText(directory)
            self.project_dir = directory
            
    def start_burst_capture(self):
        if not hasattr(self, 'confirmed_annotations'):
            self.log_status("Error: Please confirm annotations first")
            return
            
        if not self.project_dir:
            self.log_status("Error: Please select project directory")
            return
            
        # Create directory structure
        self.create_project_structure()
        
        # Set capture parameters
        self.capture_thread.set_parameters(
            self.confirmed_annotations, 
            self.class_names, 
            self.project_dir
        )
        
        # Update interval
        self.capture_thread.capture_interval = self.interval_spin.value() / 1000.0
        
        # Start capture
        self.capture_thread.start_capture()
        self.start_burst_btn.setEnabled(False)
        self.stop_burst_btn.setEnabled(True)
        self.capture_count = 0
        self.log_status("Burst capture started")
        
    def stop_burst_capture(self):
        self.capture_thread.stop_capture()
        
    def on_image_captured(self, image, filename):
        # Save image
        images_dir = os.path.join(self.project_dir, "images")
        image_path = os.path.join(images_dir, filename)
        cv2.imwrite(image_path, image)
        
        # Save annotations
        labels_dir = os.path.join(self.project_dir, "labels")
        label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in self.confirmed_annotations:
                class_id = self.class_names.index(ann['class'])
                f.write(f"{class_id} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
        
        self.capture_count += 1
        self.capture_count_label.setText(f"Images captured: {self.capture_count}")
        
    def on_capture_finished(self):
        self.start_burst_btn.setEnabled(True)
        self.stop_burst_btn.setEnabled(False)
        self.log_status(f"Burst capture finished. Total images: {self.capture_count}")
        
    def create_project_structure(self):
        # Create directories
        os.makedirs(os.path.join(self.project_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "train", "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "val", "images"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "val", "labels"), exist_ok=True)
        
    def prepare_training_data(self):
        if not self.project_dir:
            self.log_status("Error: Please select project directory")
            return
            
        images_dir = os.path.join(self.project_dir, "images")
        labels_dir = os.path.join(self.project_dir, "labels")
        
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            self.log_status("Error: No images found for training")
            return
            
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) < 2:
            self.log_status("Error: Need at least 2 images for training")
            return
            
        # Split data
        split_ratio = self.split_spin.value() / 100.0
        random.shuffle(image_files)
        split_index = int(len(image_files) * split_ratio)
        
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]
        
        # Copy files to train/val directories
        for filename in train_files:
            # Copy image
            src_img = os.path.join(images_dir, filename)
            dst_img = os.path.join(self.project_dir, "train", "images", filename)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_filename = filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            src_label = os.path.join(labels_dir, label_filename)
            dst_label = os.path.join(self.project_dir, "train", "labels", label_filename)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
                
        for filename in val_files:
            # Copy image
            src_img = os.path.join(images_dir, filename)
            dst_img = os.path.join(self.project_dir, "val", "images", filename)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_filename = filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            src_label = os.path.join(labels_dir, label_filename)
            dst_label = os.path.join(self.project_dir, "val", "labels", label_filename)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
        
        # Create YAML file
        yaml_data = {
            'path': self.project_dir,
            'train': 'train/images',
            'val': 'val/images',
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        yaml_path = os.path.join(self.project_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
            
        self.data_yaml_path = yaml_path
        self.start_training_btn.setEnabled(True)
        
        self.log_status(f"Training data prepared: {len(train_files)} train, {len(val_files)} val images")
        
    def start_training(self):
        if not hasattr(self, 'data_yaml_path'):
            self.log_status("Error: Please prepare training data first")
            return
            
        model_name = self.model_combo.currentText()
        epochs = self.epochs_spin.value()
        
        self.training_thread.set_parameters(model_name, self.data_yaml_path, epochs)
        self.training_thread.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.start_training_btn.setEnabled(False)
        
        self.log_status("Training started...")
        
    def on_training_progress(self, message):
        self.log_status(message)
        
    def on_training_finished(self, message, results_dir):
        self.progress_bar.setVisible(False)
        self.start_training_btn.setEnabled(True)
        self.log_status(message)
        
        if results_dir:
            self.results_directory = results_dir
            self.open_results_btn.setEnabled(True)
            self.log_status(f"Results saved to: {results_dir}")
            
    def open_results_folder(self):
        if hasattr(self, 'results_directory') and os.path.exists(self.results_directory):
            if sys.platform.startswith('win'):
                os.startfile(self.results_directory)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', self.results_directory])
            else:
                subprocess.run(['xdg-open', self.results_directory])
        else:
            self.log_status("Results directory not found")
            
    def log_status(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        
    def closeEvent(self, event):
        # Stop preview and cleanup
        if self.preview_active:
            self.stop_preview()
        if self.camera:
            self.camera.release()
        if self.capture_thread.isRunning():
            self.capture_thread.stop_capture()
            self.capture_thread.wait()
        if self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = YOLOTrainingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()