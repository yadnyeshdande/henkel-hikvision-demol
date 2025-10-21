#!/usr/bin/env python3
"""
YOLO11n GUI Application using PyQt5
====================================

A comprehensive GUI application for YOLO11n object detection with support for:
- Image detection
- Video processing 
- Real-time webcam detection
- Model configuration and information

Requirements:
- PyQt5
- ultralytics
- opencv-python
- pillow
- numpy

Install with: pip install PyQt5 ultralytics opencv-python pillow numpy
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                             QFileDialog, QSlider, QSpinBox, QTextEdit, 
                             QProgressBar, QGroupBox, QGridLayout, QComboBox,
                             QCheckBox, QMessageBox, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from ultralytics import YOLO
from pathlib import Path
import time
import os

class VideoThread(QThread):
    """Thread for video processing to avoid GUI freezing"""
    frame_ready = pyqtSignal(np.ndarray)
    progress_update = pyqtSignal(int, str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.output_path = ""
        self.model = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.7
        self.is_running = False
        
    def setup(self, video_path, output_path, model, conf_threshold, iou_threshold):
        self.video_path = video_path
        self.output_path = output_path
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
    def run(self):
        try:
            self.is_running = True
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                self.error_occurred.emit("Cannot open video file")
                return
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out = None
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Run inference
                results = self.model(frame, conf=self.conf_threshold, 
                                   iou=self.iou_threshold, verbose=False)
                
                annotated_frame = results[0].plot()
                
                # Add frame info
                cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.frame_ready.emit(annotated_frame)
                
                if out:
                    out.write(annotated_frame)
                
                progress = int((frame_count / total_frames) * 100)
                self.progress_update.emit(progress, f"Processing frame {frame_count}/{total_frames}")
                
                self.msleep(33)  # ~30 FPS display
            
            cap.release()
            if out:
                out.release()
                
            self.finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        self.is_running = False

class WebcamThread(QThread):
    """Thread for webcam capture"""
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    fps_update = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.camera_id = 0
        self.model = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.7
        self.is_running = False
        
    def setup(self, camera_id, model, conf_threshold, iou_threshold):
        self.camera_id = camera_id
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
    def run(self):
        try:
            self.is_running = True
            cap = cv2.VideoCapture(self.camera_id)
            
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open camera {self.camera_id}")
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            start_time = time.time()
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                results = self.model(frame, conf=self.conf_threshold,
                                   iou=self.iou_threshold, verbose=False)
                
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.frame_ready.emit(annotated_frame)
                self.fps_update.emit(current_fps)
                
                self.msleep(33)  # ~30 FPS
            
            cap.release()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        self.is_running = False

class YOLO11GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.video_thread = VideoThread()
        self.webcam_thread = WebcamThread()
        self.current_image_path = ""
        
        # Connect thread signals
        self.video_thread.frame_ready.connect(self.update_video_display)
        self.video_thread.progress_update.connect(self.update_progress)
        self.video_thread.finished.connect(self.video_processing_finished)
        self.video_thread.error_occurred.connect(self.show_error)
        
        self.webcam_thread.frame_ready.connect(self.update_webcam_display)
        self.webcam_thread.fps_update.connect(self.update_fps_display)
        self.webcam_thread.error_occurred.connect(self.show_error)
        
        self.init_ui()
        self.load_default_model()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("YOLO11n Object Detection GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create model configuration section
        model_group = self.create_model_config_section()
        main_layout.addWidget(model_group)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Create tabs
        tab_widget.addTab(self.create_image_tab(), "Image Detection")
        tab_widget.addTab(self.create_video_tab(), "Video Processing")
        tab_widget.addTab(self.create_webcam_tab(), "Live Webcam")
        tab_widget.addTab(self.create_info_tab(), "Model Information")
        
        # Status bar
        self.statusBar().showMessage("Ready - Load a model to start")
        
    def create_model_config_section(self):
        """Create model configuration section"""
        group = QGroupBox("Model Configuration")
        layout = QGridLayout()
        
        # Model path
        layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_path_label = QLabel("yolo11n.pt")
        layout.addWidget(self.model_path_label, 0, 1)
        
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(load_model_btn, 0, 2)
        
        # Confidence threshold
        layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        layout.addWidget(self.conf_slider, 1, 1)
        
        self.conf_label = QLabel("0.25")
        layout.addWidget(self.conf_label, 1, 2)
        
        # IoU threshold
        layout.addWidget(QLabel("IoU:"), 2, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(70)
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        layout.addWidget(self.iou_slider, 2, 1)
        
        self.iou_label = QLabel("0.70")
        layout.addWidget(self.iou_label, 2, 2)
        
        group.setLayout(layout)
        return group
    
    def create_image_tab(self):
        """Create image detection tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(load_btn)
        
        detect_btn = QPushButton("Detect Objects")
        detect_btn.clicked.connect(self.detect_image)
        controls_layout.addWidget(detect_btn)
        
        save_btn = QPushButton("Save Result")
        save_btn.clicked.connect(self.save_image_result)
        controls_layout.addWidget(save_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Image display
        splitter = QSplitter(Qt.Horizontal)
        
        # Original image
        original_frame = QFrame()
        original_frame.setFrameStyle(QFrame.Box)
        original_layout = QVBoxLayout(original_frame)
        original_layout.addWidget(QLabel("Original Image"))
        self.original_image_label = QLabel()
        self.original_image_label.setMinimumSize(400, 300)
        self.original_image_label.setStyleSheet("border: 1px solid gray;")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setText("No image loaded")
        original_layout.addWidget(self.original_image_label)
        
        # Result image
        result_frame = QFrame()
        result_frame.setFrameStyle(QFrame.Box)
        result_layout = QVBoxLayout(result_frame)
        result_layout.addWidget(QLabel("Detection Result"))
        self.result_image_label = QLabel()
        self.result_image_label.setMinimumSize(400, 300)
        self.result_image_label.setStyleSheet("border: 1px solid gray;")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setText("No detection performed")
        result_layout.addWidget(self.result_image_label)
        
        splitter.addWidget(original_frame)
        splitter.addWidget(result_frame)
        layout.addWidget(splitter)
        
        # Detection results text
        self.image_results_text = QTextEdit()
        self.image_results_text.setMaximumHeight(150)
        self.image_results_text.setPlaceholderText("Detection results will appear here...")
        layout.addWidget(self.image_results_text)
        
        widget.setLayout(layout)
        return widget
    
    def create_video_tab(self):
        """Create video processing tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        load_video_btn = QPushButton("Load Video")
        load_video_btn.clicked.connect(self.load_video)
        controls_layout.addWidget(load_video_btn)
        
        self.process_video_btn = QPushButton("Process Video")
        self.process_video_btn.clicked.connect(self.process_video)
        controls_layout.addWidget(self.process_video_btn)
        
        self.stop_video_btn = QPushButton("Stop")
        self.stop_video_btn.clicked.connect(self.stop_video_processing)
        self.stop_video_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_video_btn)
        
        self.save_video_checkbox = QCheckBox("Save Output")
        controls_layout.addWidget(self.save_video_checkbox)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.video_progress = QProgressBar()
        layout.addWidget(self.video_progress)
        
        # Video display
        self.video_display_label = QLabel()
        self.video_display_label.setMinimumSize(640, 480)
        self.video_display_label.setStyleSheet("border: 1px solid gray;")
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setText("No video loaded")
        layout.addWidget(self.video_display_label)
        
        # Status
        self.video_status_label = QLabel("Ready to process video")
        layout.addWidget(self.video_status_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_webcam_tab(self):
        """Create webcam tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Camera selection
        controls_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        for i in range(5):  # Check for up to 5 cameras
            self.camera_combo.addItem(f"Camera {i}")
        controls_layout.addWidget(self.camera_combo)
        
        self.start_webcam_btn = QPushButton("Start Webcam")
        self.start_webcam_btn.clicked.connect(self.start_webcam)
        controls_layout.addWidget(self.start_webcam_btn)
        
        self.stop_webcam_btn = QPushButton("Stop Webcam")
        self.stop_webcam_btn.clicked.connect(self.stop_webcam)
        self.stop_webcam_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_webcam_btn)
        
        screenshot_btn = QPushButton("Screenshot")
        screenshot_btn.clicked.connect(self.take_screenshot)
        controls_layout.addWidget(screenshot_btn)
        
        controls_layout.addStretch()
        
        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 12, QFont.Bold))
        controls_layout.addWidget(self.fps_label)
        
        layout.addLayout(controls_layout)
        
        # Webcam display
        self.webcam_display_label = QLabel()
        self.webcam_display_label.setMinimumSize(640, 480)
        self.webcam_display_label.setStyleSheet("border: 1px solid gray;")
        self.webcam_display_label.setAlignment(Qt.AlignCenter)
        self.webcam_display_label.setText("Click 'Start Webcam' to begin")
        layout.addWidget(self.webcam_display_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_info_tab(self):
        """Create model information tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        refresh_btn = QPushButton("Refresh Model Info")
        refresh_btn.clicked.connect(self.refresh_model_info)
        layout.addWidget(refresh_btn)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        widget.setLayout(layout)
        return widget
    
    def load_default_model(self):
        """Load default YOLO11n model"""
        try:
            self.statusBar().showMessage("Loading default YOLO11n model...")
            self.model = YOLO('yolo11n.pt')
            self.statusBar().showMessage("YOLO11n model loaded successfully")
            self.refresh_model_info()
        except Exception as e:
            self.show_error(f"Failed to load default model: {str(e)}")
    
    def load_model(self):
        """Load custom model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "Model files (*.pt *.onnx)")
        
        if file_path:
            try:
                self.statusBar().showMessage("Loading model...")
                self.model = YOLO(file_path)
                self.model_path_label.setText(os.path.basename(file_path))
                self.statusBar().showMessage(f"Model loaded: {os.path.basename(file_path)}")
                self.refresh_model_info()
            except Exception as e:
                self.show_error(f"Failed to load model: {str(e)}")
    
    def load_image(self):
        """Load image for detection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff)")
        
        if file_path:
            self.current_image_path = file_path
            
            # Display original image
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.original_image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.original_image_label.setPixmap(scaled_pixmap)
            
            # Clear previous results
            self.result_image_label.setText("Click 'Detect Objects' to analyze")
            self.image_results_text.clear()
    
    def detect_image(self):
        """Perform object detection on loaded image"""
        if not self.model:
            self.show_error("Please load a model first")
            return
        
        if not self.current_image_path:
            self.show_error("Please load an image first")
            return
        
        try:
            self.statusBar().showMessage("Processing image...")
            
            # Get current settings
            conf_threshold = self.conf_slider.value() / 100.0
            iou_threshold = self.iou_slider.value() / 100.0
            
            # Run inference
            results = self.model(self.current_image_path, 
                               conf=conf_threshold, iou=iou_threshold)
            result = results[0]
            
            # Display annotated image
            annotated_img = result.plot()
            height, width, channel = annotated_img.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotated_img.data, width, height, 
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.result_image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.result_image_label.setPixmap(scaled_pixmap)
            
            # Display detection results
            detections = result.boxes
            results_text = f"Image: {os.path.basename(self.current_image_path)}\n"
            results_text += f"Confidence: {conf_threshold}, IoU: {iou_threshold}\n\n"
            
            if detections is not None and len(detections) > 0:
                results_text += f"Found {len(detections)} objects:\n"
                for i, box in enumerate(detections):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    bbox = box.xyxy[0].tolist()
                    results_text += f"{i+1}. {class_name}: {confidence:.2f} "
                    results_text += f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]\n"
            else:
                results_text += "No objects detected"
            
            self.image_results_text.setText(results_text)
            self.statusBar().showMessage("Image processing completed")
            
        except Exception as e:
            self.show_error(f"Detection failed: {str(e)}")
    
    def save_image_result(self):
        """Save detection result image"""
        if self.result_image_label.pixmap() is None:
            self.show_error("No detection result to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "", 
            "Image files (*.jpg *.png *.bmp)")
        
        if file_path:
            self.result_image_label.pixmap().save(file_path)
            self.statusBar().showMessage(f"Result saved: {file_path}")
    
    def load_video(self):
        """Load video for processing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", 
            "Video files (*.mp4 *.avi *.mov *.mkv)")
        
        if file_path:
            self.video_path = file_path
            self.video_status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            
            # Reset progress
            self.video_progress.setValue(0)
            self.video_display_label.setText("Ready to process video")
    
    def process_video(self):
        """Start video processing"""
        if not self.model:
            self.show_error("Please load a model first")
            return
        
        if not hasattr(self, 'video_path'):
            self.show_error("Please load a video first")
            return
        
        # Get output path if saving
        output_path = ""
        if self.save_video_checkbox.isChecked():
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Processed Video", "", 
                "Video files (*.mp4)")
            if not output_path:
                return
        
        # Setup and start video thread
        conf_threshold = self.conf_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0
        
        self.video_thread.setup(
            self.video_path, output_path, self.model, 
            conf_threshold, iou_threshold)
        
        self.video_thread.start()
        
        # Update UI
        self.process_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(True)
        self.video_status_label.setText("Processing video...")
    
    def stop_video_processing(self):
        """Stop video processing"""
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        self.process_video_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(False)
        self.video_status_label.setText("Processing stopped")
    
    def start_webcam(self):
        """Start webcam detection"""
        if not self.model:
            self.show_error("Please load a model first")
            return
        
        camera_id = self.camera_combo.currentIndex()
        conf_threshold = self.conf_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0
        
        self.webcam_thread.setup(camera_id, self.model, conf_threshold, iou_threshold)
        self.webcam_thread.start()
        
        # Update UI
        self.start_webcam_btn.setEnabled(False)
        self.stop_webcam_btn.setEnabled(True)
    
    def stop_webcam(self):
        """Stop webcam detection"""
        if self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.webcam_thread.wait()
        
        self.start_webcam_btn.setEnabled(True)
        self.stop_webcam_btn.setEnabled(False)
        self.webcam_display_label.setText("Webcam stopped")
        self.fps_label.setText("FPS: 0.0")
    
    def take_screenshot(self):
        """Take screenshot of current webcam frame"""
        if self.webcam_display_label.pixmap() is None:
            self.show_error("No webcam frame to save")
            return
        
        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.jpg"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", filename, 
            "Image files (*.jpg *.png)")
        
        if file_path:
            self.webcam_display_label.pixmap().save(file_path)
            self.statusBar().showMessage(f"Screenshot saved: {file_path}")
    
    def refresh_model_info(self):
        """Refresh model information display"""
        if not self.model:
            self.info_text.setText("No model loaded")
            return
        
        info_text = "YOLO11n Model Information\n"
        info_text += "=" * 50 + "\n\n"
        
        try:
            info_text += f"Model type: {type(self.model).__name__}\n"
            info_text += f"Number of classes: {len(self.model.names)}\n"
            info_text += f"Confidence threshold: {self.conf_slider.value() / 100.0:.2f}\n"
            info_text += f"IoU threshold: {self.iou_slider.value() / 100.0:.2f}\n\n"
            
            info_text += "Supported classes:\n"
            info_text += "-" * 30 + "\n"
            
            for i, class_name in self.model.names.items():
                info_text += f"{i:2d}: {class_name}\n"
                
        except Exception as e:
            info_text += f"Error retrieving model info: {str(e)}"
        
        self.info_text.setText(info_text)
    
    # Slot methods for thread communication
    @pyqtSlot(np.ndarray)
    def update_video_display(self, frame):
        """Update video display with new frame"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_display_label.setPixmap(scaled_pixmap)
    
    @pyqtSlot(np.ndarray)
    def update_webcam_display(self, frame):
        """Update webcam display with new frame"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.webcam_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.webcam_display_label.setPixmap(scaled_pixmap)
    
    @pyqtSlot(int, str)
    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.video_progress.setValue(value)
        self.video_status_label.setText(message)
    
    @pyqtSlot()
    def video_processing_finished(self):
        """Handle video processing completion"""
        self.process_video_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(False)
        self.video_status_label.setText("Video processing completed")
        self.statusBar().showMessage("Video processing finished")
    
    @pyqtSlot(float)
    def update_fps_display(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @pyqtSlot(str)
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.statusBar().showMessage(f"Error: {message}")
    
    def update_conf_label(self):
        """Update confidence threshold label"""
        value = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"{value:.2f}")
    
    def update_iou_label(self):
        """Update IoU threshold label"""
        value = self.iou_slider.value() / 100.0
        self.iou_label.setText(f"{value:.2f}")
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop any running threads
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        if self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.webcam_thread.wait()
        
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO11n Object Detection GUI")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = YOLO11GUI()
    window.show()
    
    # Handle application exit
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass

if __name__ == "__main__":
    main()

# Example usage and features:
"""
YOLO11n GUI Application Features:
=================================

1. MODEL CONFIGURATION:
   - Load custom YOLO models (.pt, .onnx)
   - Adjust confidence threshold (0.01 - 1.00)
   - Adjust IoU threshold for NMS (0.01 - 1.00)
   - Real-time parameter updates

2. IMAGE DETECTION:
   - Load images (JPG, PNG, BMP, TIFF)
   - Side-by-side original and result display
   - Detailed detection results with bounding boxes
   - Save annotated results

3. VIDEO PROCESSING:
   - Load and process video files (MP4, AVI, MOV, MKV)
   - Real-time progress tracking
   - Optional output video saving
   - Frame-by-frame analysis display

4. LIVE WEBCAM:
   - Multi-camera support (up to 5 cameras)
   - Real-time FPS monitoring
   - Screenshot functionality
   - Live object detection overlay

5. MODEL INFORMATION:
   - Complete class list display
   - Model statistics and parameters
   - Real-time configuration updates

Installation Requirements:
-------------------------
pip install PyQt5 ultralytics opencv-python pillow numpy

Key Features:
------------
- Threaded processing to prevent GUI freezing
- Professional tabbed interface
- Real-time parameter adjustment
- Progress tracking and status updates
- Error handling and user feedback
- Screenshot and result saving capabilities
- Multi-format support for images and videos

Usage Tips:
----------
- The application automatically downloads YOLO11n on first run
- Use the sliders to adjust detection sensitivity in real-time
- All processing runs in separate threads for smooth GUI operation
- Screenshots and results can be saved in multiple formats
- The model information tab shows all supported object classes
"""