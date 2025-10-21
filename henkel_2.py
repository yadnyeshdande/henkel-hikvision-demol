import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import json
import os
import pyhid_usb_relay
from ultralytics import YOLO
import threading

class DrawingWidget(QLabel):
    """Custom widget for drawing boundaries like Microsoft Paint"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("border: 2px solid #333; background-color: white;")
        self.setAlignment(Qt.AlignCenter)
        self.image = None
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rectangles = []
        self.boundaries = []
        self.current_rect_preview = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        
        # Colors and styling for different boundary types
        self.blue_box_color = QColor(0, 0, 255)  # Blue for blue boxes
        self.yellow_mark_color = QColor(255, 255, 0)  # Yellow for yellow marks
        self.preview_color = QColor(0, 255, 0, 100)  # Semi-transparent green
        self.line_width = 3
        
        # Track current boundary type
        self.current_boundary_type = "blue_box"  # Start with blue boxes

    def set_image(self, cv_image):
        """Set the image for drawing boundaries"""
        self.cv_image = cv_image  # Save for future resizing
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.original_pixmap = pixmap
        self.image = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Calculate scaling factors for coordinate conversion
        self.scale_x = pixmap.width() / self.image.width()
        self.scale_y = pixmap.height() / self.image.height()
        
        # Calculate offset to center the image
        self.offset_x = (self.width() - self.image.width()) // 2
        self.offset_y = (self.height() - self.image.height()) // 2
        
        self.clear_boundaries()
        self.update_display()

    def resizeEvent(self, event):
        """Handle widget resizing to keep image centered"""
        if hasattr(self, 'cv_image') and self.cv_image is not None:
            self.set_image(self.cv_image)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image and len(self.rectangles) < 9:
            # Convert widget coordinates to image coordinates
            img_pos = self.widget_to_image_coords(event.pos())
            if img_pos:
                self.drawing = True
                self.start_point = img_pos
                self.setCursor(Qt.CrossCursor)

    def mouseMoveEvent(self, event):
        if self.drawing and self.image:
            img_pos = self.widget_to_image_coords(event.pos())
            if img_pos:
                self.end_point = img_pos
                self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and self.image:
            img_pos = self.widget_to_image_coords(event.pos())
            if img_pos:
                self.end_point = img_pos
                self.finish_rectangle()
                self.drawing = False
                self.setCursor(Qt.CrossCursor)

    def widget_to_image_coords(self, widget_point):
        """Convert widget coordinates to image coordinates"""
        if not self.image:
            return None
        
        # Check if point is within image bounds
        img_rect = QRect(self.offset_x, self.offset_y, self.image.width(), self.image.height())
        if not img_rect.contains(widget_point):
            return None
        
        # Convert to image coordinates
        x = widget_point.x() - self.offset_x
        y = widget_point.y() - self.offset_y
        return QPoint(x, y)

    def image_to_widget_coords(self, image_point):
        """Convert image coordinates to widget coordinates"""
        return QPoint(
            image_point.x() + self.offset_x,
            image_point.y() + self.offset_y
        )

    def finish_rectangle(self):
        """Finish drawing current rectangle"""
        if self.start_point and self.end_point:
            # Ensure we have top-left and bottom-right points
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())
            
            # Only add if rectangle has meaningful size
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                rect = QRect(x1, y1, x2 - x1, y2 - y1)
                self.rectangles.append(rect)
                
                # Determine boundary type based on current count
                current_count = len(self.rectangles)
                boundary_type = "blue_box" if current_count <= 3 else "yellow_mark"
                
                # Convert to original image coordinates for saving
                boundary = {
                    'x1': x1 * self.scale_x,
                    'y1': y1 * self.scale_y,
                    'x2': x2 * self.scale_x,
                    'y2': y2 * self.scale_y,
                    'type': boundary_type
                }
                self.boundaries.append(boundary)
                self.update_display()

    def update_display(self):
        """Update the display with current rectangles"""
        if not self.image:
            return
        
        # Create a copy of the image to draw on
        display_pixmap = self.image.copy()
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw completed rectangles
        for i, rect in enumerate(self.rectangles):
            # Choose color based on boundary type
            if i < 3:  # First 3 are blue boxes
                color = self.blue_box_color
                label = f"Blue Box {i + 1}"
            else:  # Next 6 are yellow marks
                color = self.yellow_mark_color
                label = f"Yellow Mark {i - 2}"
            
            pen = QPen(color, self.line_width)
            painter.setPen(pen)
            painter.drawRect(rect)
            
            # Draw area label
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            label_pos = QPoint(rect.x() + 5, rect.y() - 5)
            painter.drawText(label_pos, label)
        
        # Draw current rectangle being drawn
        if self.drawing and self.start_point and self.end_point:
            pen = QPen(self.preview_color, self.line_width)
            painter.setPen(pen)
            current_rect = QRect(
                min(self.start_point.x(), self.end_point.x()),
                min(self.start_point.y(), self.end_point.y()),
                abs(self.end_point.x() - self.start_point.x()),
                abs(self.end_point.y() - self.start_point.y())
            )
            painter.drawRect(current_rect)
        
        painter.end()
        
        # Center the image in the widget
        final_pixmap = QPixmap(self.size())
        final_pixmap.fill(Qt.white)
        final_painter = QPainter(final_pixmap)
        final_painter.drawPixmap(self.offset_x, self.offset_y, display_pixmap)
        final_painter.end()
        
        self.setPixmap(final_pixmap)

    def clear_boundaries(self):
        """Clear all boundaries"""
        self.rectangles.clear()
        self.boundaries.clear()
        self.drawing = False
        if self.image:
            self.update_display()

    def get_instruction_text(self):
        """Get current instruction text"""
        completed = len(self.rectangles)
        if completed < 3:
            return f"Draw Blue Box boundary {completed + 1} of 3 - Click and drag to create rectangle"
        elif completed < 9:
            yellow_count = completed - 3 + 1
            return f"Draw Yellow Mark boundary {yellow_count} of 6 - Click and drag to create rectangle"
        else:
            return "All 9 boundaries completed! Click 'Save Boundaries' to proceed."

class TrainingPage(QWidget):
    """Page for defining detection boundaries"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training - Define Container Boundaries")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Instructions panel
        instructions = QLabel(
            "Instructions:\n"
            "1. First draw 3 boundaries for BLUE BOXES\n"
            "2. Then draw 6 boundaries for YELLOW MARKS\n"
            "3. Total: 9 boundaries required"
        )
        instructions.setStyleSheet(
            "background-color: #E1F5FE; padding: 10px; border-radius: 5px; "
            "font-size: 12px; color: #0277BD; border: 1px solid #0288D1;"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        camera_controls.addWidget(QLabel("Camera:"))
        camera_controls.addWidget(self.camera_combo)
        
        self.start_camera_btn = QPushButton("Start Camera Preview")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        camera_controls.addWidget(self.start_camera_btn)
        
        self.capture_btn = QPushButton("📷 Capture Photo")
        self.capture_btn.clicked.connect(self.capture_frame)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        camera_controls.addWidget(self.capture_btn)
        
        layout.addLayout(camera_controls)
        
        # Large area for camera preview and boundary drawing
        self.image_area = QStackedWidget()
        
        self.camera_label = QLabel("📹 Start camera to see live preview")
        self.camera_label.setMinimumSize(900, 600)
        self.camera_label.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5; font-size: 16px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.image_area.addWidget(self.camera_label)
        
        self.drawing_widget = DrawingWidget()
        self.drawing_widget.setMinimumSize(900, 600)
        self.image_area.addWidget(self.drawing_widget)
        
        layout.addWidget(self.image_area)
        
        # Instructions
        self.instruction_label = QLabel("📷 Click 'Capture Photo' first to start drawing boundaries")
        self.instruction_label.setStyleSheet(
            "background-color: #E3F2FD; padding: 10px; border-radius: 5px; "
            "font-size: 12px; color: #1976D2;"
        )
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑️ Clear All Boundaries")
        self.clear_btn.clicked.connect(self.clear_boundaries)
        self.clear_btn.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
        button_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("💾 Save Boundaries")
        self.save_btn.clicked.connect(self.save_boundaries)
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("✅ Ready - Follow the steps above to define boundaries")
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: #E8F5E8; color: #2E7D32; "
            "border: 1px solid #4CAF50; border-radius: 5px; font-weight: bold;"
        )
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def refresh_cameras(self):
        """Refresh available cameras"""
        self.camera_combo.clear()
        for i in range(4):  # Check first 4 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()

    def start_camera(self):
        """Start camera preview"""
        camera_index = self.camera_combo.currentIndex()
        if self.camera is not None:
            self.camera.release()
        
        self.camera = cv2.VideoCapture(camera_index)
        if self.camera.isOpened():
            self.image_area.setCurrentWidget(self.camera_label)  # Show camera preview
            self.timer.start(30)  # Update every 30ms
            self.start_camera_btn.setText("⏹️ Stop Camera")
            self.start_camera_btn.setStyleSheet("background-color: #F44336; color: white; padding: 8px;")
            self.start_camera_btn.clicked.disconnect()
            self.start_camera_btn.clicked.connect(self.stop_camera)
            self.capture_btn.setEnabled(True)
            self.status_label.setText(f"📹 Camera {camera_index} is running - Ready to capture photo")
            self.status_label.setStyleSheet(
                "padding: 15px; background-color: #E8F5E8; color: #2E7D32; "
                "border: 1px solid #4CAF50; border-radius: 5px; font-weight: bold;"
            )
        else:
            QMessageBox.warning(self, "Error", "Cannot access camera!")

    def stop_camera(self):
        """Stop camera preview"""
        self.timer.stop()
        if self.camera:
            self.camera.release()
        
        self.camera_label.clear()
        self.camera_label.setText("📹 Start camera to see live preview")
        self.start_camera_btn.setText("Start Camera Preview")
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.start_camera_btn.clicked.disconnect()
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.capture_btn.setEnabled(False)
        self.status_label.setText("⏹️ Camera stopped - Start camera to continue")
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: #FFEBEE; color: #C62828; "
            "border: 1px solid #F44336; border-radius: 5px; font-weight: bold;"
        )
        self.image_area.setCurrentWidget(self.camera_label)  # Show camera preview by default

    def update_frame(self):
        """Update camera preview"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.camera_label.setPixmap(scaled_pixmap)
                self.current_frame = frame.copy()

    def capture_frame(self):
        """Capture current frame for boundary training"""
        if hasattr(self, 'current_frame'):
            self.stop_camera()
            self.drawing_widget.set_image(self.current_frame)
            self.captured_frame = self.current_frame.copy()
            self.image_area.setCurrentWidget(self.drawing_widget)  # Switch to drawing widget
            self.update_instruction_text()
            self.status_label.setText("📷 Photo captured! Now draw 3 blue box boundaries + 6 yellow mark boundaries")
            self.status_label.setStyleSheet(
                "padding: 15px; background-color: #FFF3E0; color: #E65100; "
                "border: 1px solid #FF9800; border-radius: 5px; font-weight: bold;"
            )
        else:
            QMessageBox.warning(self, "Error", "No frame available to capture!")

    def update_instruction_text(self):
        """Update instruction text based on drawing progress"""
        instruction_text = self.drawing_widget.get_instruction_text()
        self.instruction_label.setText(f"✏️ {instruction_text}")

    def clear_boundaries(self):
        """Clear all boundaries"""
        self.drawing_widget.clear_boundaries()
        self.update_instruction_text()
        self.status_label.setText("🗑️ Boundaries cleared - Draw new boundaries")
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: #FFF3E0; color: #E65100; "
            "border: 1px solid #FF9800; border-radius: 5px; font-weight: bold;"
        )

    def save_boundaries(self):
        """Save boundaries to file"""
        if len(self.drawing_widget.boundaries) != 9:
            QMessageBox.warning(self, "Error", "Please draw exactly 9 boundaries first!\n(3 blue boxes + 6 yellow marks)")
            return
        
        # Separate boundaries by type
        blue_box_boundaries = [b for b in self.drawing_widget.boundaries if b['type'] == 'blue_box']
        yellow_mark_boundaries = [b for b in self.drawing_widget.boundaries if b['type'] == 'yellow_mark']
        
        # Save boundaries and reference frame
        data = {
            'blue_box_boundaries': blue_box_boundaries,
            'yellow_mark_boundaries': yellow_mark_boundaries,
            'all_boundaries': self.drawing_widget.boundaries,
            'frame_shape': self.captured_frame.shape
        }
        
        try:
            with open('boundaries.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            cv2.imwrite('reference_frame.jpg', self.captured_frame)
            
            self.status_label.setText("💾 Boundaries saved successfully! Ready for detection.")
            self.status_label.setStyleSheet(
                "padding: 15px; background-color: #E8F5E8; color: #2E7D32; "
                "border: 1px solid #4CAF50; border-radius: 5px; font-weight: bold;"
            )
            
            QMessageBox.information(
                self, "Success",
                f"✅ Boundaries saved to boundaries.json\n"
                f"📷 Reference frame saved\n"
                f"🔵 Blue boxes: {len(blue_box_boundaries)}\n"
                f"🟡 Yellow marks: {len(yellow_mark_boundaries)}\n\n"
                f"You can now go to Detection!"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save boundaries: {str(e)}")

    def closeEvent(self, event):
        """Clean up when closing"""
        if self.camera:
            self.camera.release()
        event.accept()

class DetectionPage(QWidget):
    """Page for real-time blue oil container detection"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_objects)
        self.blue_box_boundaries = []
        self.yellow_mark_boundaries = []
        self.all_boundaries = []
        self.yolo_model = None
        self.last_frame_time = None
        self.current_fps = 0
        self.load_boundaries()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Detection - Blue Container & Yellow Mark Monitoring")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        controls_layout.addWidget(QLabel("Camera:"))
        controls_layout.addWidget(self.camera_combo)
        
        # Load model button
        self.model_path = "yolo11n.pt"
        self.load_model_btn = QPushButton("Load Custom Model")
        self.load_model_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_model_btn)
        
        # Start detection button
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        controls_layout.addWidget(self.start_btn)
        
        layout.addLayout(controls_layout)
        
        # Detection display
        self.detection_label = QLabel("Detection View")
        self.detection_label.setMinimumSize(800, 600)
        self.detection_label.setStyleSheet("border: 1px solid black;")
        self.detection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detection_label)
        
        # Status indicators - Blue Boxes
        blue_status_layout = QHBoxLayout()
        blue_label = QLabel("Blue Box Areas:")
        blue_label.setStyleSheet("font-weight: bold; color: #0000FF;")
        blue_status_layout.addWidget(blue_label)
        
        self.blue_status_labels = []
        for i in range(3):
            status_label = QLabel(f"Blue Box {i + 1}: Unknown")
            status_label.setStyleSheet(
                "padding: 10px; font-size: 14px; font-weight: bold; "
                "background-color: #cccccc; border-radius: 5px;"
            )
            status_label.setAlignment(Qt.AlignCenter)
            blue_status_layout.addWidget(status_label)
            self.blue_status_labels.append(status_label)
        
        layout.addLayout(blue_status_layout)
        
        # Status indicators - Yellow Marks
        yellow_status_layout = QHBoxLayout()
        yellow_label = QLabel("Yellow Mark Areas:")
        yellow_label.setStyleSheet("font-weight: bold; color: #FFD700;")
        yellow_status_layout.addWidget(yellow_label)
        
        # Create two rows for yellow marks (3 each)
        yellow_row1_layout = QHBoxLayout()
        yellow_row2_layout = QHBoxLayout()
        
        self.yellow_status_labels = []
        for i in range(6):
            status_label = QLabel(f"Yellow {i + 1}: Unknown")
            status_label.setStyleSheet(
                "padding: 8px; font-size: 12px; font-weight: bold; "
                "background-color: #cccccc; border-radius: 5px;"
            )
            status_label.setAlignment(Qt.AlignCenter)
            
            if i < 3:
                yellow_row1_layout.addWidget(status_label)
            else:
                yellow_row2_layout.addWidget(status_label)
            
            self.yellow_status_labels.append(status_label)
        
        layout.addLayout(yellow_status_layout)
        layout.addLayout(yellow_row1_layout)
        layout.addLayout(yellow_row2_layout)
        
        # Overall status
        self.overall_status = QLabel("Overall Status: Not Running")
        self.overall_status.setStyleSheet(
            "padding: 15px; font-size: 16px; font-weight: bold; "
            "background-color: #f0f0f0; border-radius: 5px;"
        )
        self.overall_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.overall_status)
        
        self.setLayout(layout)

    def refresh_cameras(self):
        """Refresh available cameras"""
        self.camera_combo.clear()
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()

    def load_boundaries(self):
        """Load saved boundaries"""
        try:
            if os.path.exists('boundaries.json'):
                with open('boundaries.json', 'r') as f:
                    data = json.load(f)
                
                # Load boundaries by type
                self.blue_box_boundaries = data.get('blue_box_boundaries', [])
                self.yellow_mark_boundaries = data.get('yellow_mark_boundaries', [])
                self.all_boundaries = data.get('all_boundaries', [])
                
                # Fallback for older format
                if not self.all_boundaries and 'boundaries' in data:
                    self.all_boundaries = data['boundaries']
                    # Assume first 3 are blue boxes, rest are yellow marks
                    self.blue_box_boundaries = self.all_boundaries[:3]
                    self.yellow_mark_boundaries = self.all_boundaries[3:]
                
            else:
                QMessageBox.warning(self, "Warning", "No boundaries file found. Please train first!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load boundaries: {str(e)}")

    def load_model(self):
        """Load YOLO model from file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom YOLO Model", "", "YOLO Model Files (*.pt *.onnx *.engine);;All Files (*)"
        )
        
        if file_path:
            try:
                self.yolo_model = YOLO(file_path)
                self.model_path = file_path
                self.start_btn.setEnabled(True)
                QMessageBox.information(self, "Success", f"Custom YOLO model loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def start_detection(self):
        """Start/stop detection"""
        if self.timer.isActive():
            self.stop_detection()
        else:
            if not self.yolo_model:
                QMessageBox.warning(self, "Error", "Please load a custom YOLO model first!")
                return
            
            if not self.all_boundaries:
                QMessageBox.warning(self, "Error", "No boundaries defined. Please train first!")
                return
            
            camera_index = self.camera_combo.currentIndex()
            self.camera = cv2.VideoCapture(camera_index)
            
            if self.camera.isOpened():
                self.timer.start(100)  # Update every 100ms
                self.start_btn.setText("Stop Detection")
                self.overall_status.setText("Detection Running...")
            else:
                QMessageBox.warning(self, "Error", "Cannot access camera!")

    def stop_detection(self):
        """Stop detection"""
        self.timer.stop()
        if self.camera:
            self.camera.release()
        
        self.start_btn.setText("Start Detection")
        self.overall_status.setText("Detection Stopped")
        self.detection_label.clear()
        self.detection_label.setText("Detection View")
        
        # Reset blue box status labels
        for label in self.blue_status_labels:
            label.setText(label.text().split(':')[0] + ": Unknown")
            label.setStyleSheet(
                "padding: 10px; font-size: 14px; font-weight: bold; "
                "background-color: #cccccc; border-radius: 5px;"
            )
        
        # Reset yellow mark status labels
        for label in self.yellow_status_labels:
            label.setText(label.text().split(':')[0] + ": Unknown")
            label.setStyleSheet(
                "padding: 8px; font-size: 12px; font-weight: bold; "
                "background-color: #cccccc; border-radius: 5px;"
            )

    def detect_objects(self):
        """Perform object detection on current frame"""
        if not self.camera or not self.camera.isOpened():
            return
        
        ret, frame = self.camera.read()
        if not ret:
            return
        
        try:
            # Run YOLO detection with custom model
            results = self.yolo_model(frame, verbose=False)
            
            # Extract detected objects (assuming custom classes: 0=blue_box, 1=yellow_mark)
            blue_boxes = []
            yellow_marks = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if confidence > 0.5:  # Confidence threshold
                            detected_obj = {
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                            }
                            
                            if class_id == 0:  # blue_box class
                                blue_boxes.append(detected_obj)
                            elif class_id == 1:  # yellow_mark class
                                yellow_marks.append(detected_obj)
            
            # Check blue box areas
            blue_status = []
            for i, boundary in enumerate(self.blue_box_boundaries):
                objects_in_area = self.check_objects_in_boundary(blue_boxes, boundary)
                has_object = len(objects_in_area) > 0
                blue_status.append(has_object)
                
                # Update blue box status label
                status_text = f"Blue Box {i + 1}: {'✓ OK' if has_object else '✗ Empty'}"
                color = "#4CAF50" if has_object else "#F44336"  # Green or Red
                self.blue_status_labels[i].setText(status_text)
                self.blue_status_labels[i].setStyleSheet(
                    f"padding: 10px; font-size: 14px; font-weight: bold; "
                    f"background-color: {color}; color: white; border-radius: 5px;"
                )
            
            # Check yellow mark areas
            yellow_status = []
            for i, boundary in enumerate(self.yellow_mark_boundaries):
                objects_in_area = self.check_objects_in_boundary(yellow_marks, boundary)
                has_object = len(objects_in_area) > 0
                yellow_status.append(has_object)
                
                # Update yellow mark status label
                status_text = f"Yellow {i + 1}: {'✓ OK' if has_object else '✗ Missing'}"
                color = "#FFD700" if has_object else "#F44336"  # Yellow or Red
                text_color = "black" if has_object else "white"
                self.yellow_status_labels[i].setText(status_text)
                self.yellow_status_labels[i].setStyleSheet(
                    f"padding: 8px; font-size: 12px; font-weight: bold; "
                    f"background-color: {color}; color: {text_color}; border-radius: 5px;"
                )
            
            # Draw visualization
            self.draw_detection_frame(frame, blue_boxes, yellow_marks, blue_status, yellow_status)
            
            # Update overall status
            all_blue_ok = all(blue_status)
            all_yellow_ok = all(yellow_status)
            overall_ok = all_blue_ok and all_yellow_ok
            
            if overall_ok:
                overall_text = "Overall Status: ✅ All Areas OK"
                overall_color = "#4CAF50"
            else:
                issues = []
                if not all_blue_ok:
                    issues.append("Blue Boxes")
                if not all_yellow_ok:
                    issues.append("Yellow Marks")
                overall_text = f"Overall Status: ⚠️ Issues in {', '.join(issues)}"
                overall_color = "#F44336"
            
            self.overall_status.setText(overall_text)
            self.overall_status.setStyleSheet(
                f"padding: 15px; font-size: 16px; font-weight: bold; "
                f"background-color: {overall_color}; color: white; border-radius: 5px;"
            )
            
            # --- Relay control ---
            try:
                relay = pyhid_usb_relay.find()
                if overall_ok:
                    relay.set_state(1, True)   # Turn ON Relay 1
                    relay.set_state(2, False)  # Turn OFF Relay 2
                else:
                    relay.set_state(1, False)  # Turn OFF Relay 1
                    relay.set_state(2, True)   # Turn ON Relay 2
            except Exception as e:
                print(f"Relay error: {e}")
                
        except Exception as e:
            print(f"Detection error: {e}")

    def check_objects_in_boundary(self, objects, boundary):
        """Check if objects are within boundary"""
        objects_in_boundary = []
        for obj in objects:
            center_x, center_y = obj['center']
            
            if (boundary['x1'] <= center_x <= boundary['x2'] and
                boundary['y1'] <= center_y <= boundary['y2']):
                objects_in_boundary.append(obj)
        
        return objects_in_boundary

    def draw_detection_frame(self, frame, blue_boxes, yellow_marks, blue_status, yellow_status):
        """Draw detection results on frame"""
        import time
        display_frame = frame.copy()
        
        # Draw blue box boundary rectangles
        for i, boundary in enumerate(self.blue_box_boundaries):
            color = (0, 255, 0) if blue_status[i] else (0, 0, 255)  # Green if OK, Red if not
            cv2.rectangle(
                display_frame,
                (int(boundary['x1']), int(boundary['y1'])),
                (int(boundary['x2']), int(boundary['y2'])),
                color, 2
            )
            # Add area label
            cv2.putText(
                display_frame,
                f"Blue Box {i + 1}",
                (int(boundary['x1']), int(boundary['y1']) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        
        # Draw yellow mark boundary rectangles
        for i, boundary in enumerate(self.yellow_mark_boundaries):
            color = (0, 255, 255) if yellow_status[i] else (0, 0, 255)  # Yellow if OK, Red if not
            cv2.rectangle(
                display_frame,
                (int(boundary['x1']), int(boundary['y1'])),
                (int(boundary['x2']), int(boundary['y2'])),
                color, 2
            )
            # Add area label
            cv2.putText(
                display_frame,
                f"Yellow {i + 1}",
                (int(boundary['x1']), int(boundary['y1']) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        # Draw detected blue boxes
        for obj in blue_boxes:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            label = f"Blue Box: {obj['confidence']:.2f}"
            cv2.putText(
                display_frame, label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )
        
        # Draw detected yellow marks
        for obj in yellow_marks:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            label = f"Yellow Mark: {obj['confidence']:.2f}"
            cv2.putText(
                display_frame, label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
        
        # --- FPS calculation and display ---
        now = time.time()
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
            if dt > 0:
                self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt) if self.current_fps else 1.0 / dt
        self.last_frame_time = now
        
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(
            display_frame, fps_text,
            (10, 30),  # Top-left corner
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )
        
        # Convert to Qt format and display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.detection_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.detection_label.setPixmap(scaled_pixmap)

    def browse_model_file(self):
        """Open file dialog to select YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model File", "", "YOLO Model Files (*.pt *.onnx *.engine);;All Files (*)"
        )
        if file_path:
            self.model_path = file_path
            if self.model_combo.findText(file_path) == -1:
                self.model_combo.addItem(file_path)
            self.model_combo.setCurrentText(file_path)
            # This will trigger on_model_selected and load the model

    def on_model_selected(self):
        model_name = self.model_combo.currentText()
        if model_name and model_name != 'Select Model File...':
            self.load_yolo_model()

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Blue Container & Yellow Mark Detection System - PSB Automation")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add pages
        self.training_page = TrainingPage()
        self.detection_page = DetectionPage()
        
        self.tab_widget.addTab(self.training_page, "🎯 Training")
        self.tab_widget.addTab(self.detection_page, "🔍 Detection")
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Add menu bar
        self.create_menu_bar()
        
        # Add status bar
        self.statusBar().showMessage("Ready - Start with Training to define 9 boundaries (3 blue boxes + 6 yellow marks)")

    def on_tab_changed(self, index):
        # If Detection tab is selected, reload boundaries
        if self.tab_widget.widget(index) is self.detection_page:
            self.detection_page.load_boundaries()

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About",
            "Blue Container & Yellow Mark Detection System\n\n"
            "This application uses custom trained YOLO models to detect:\n"
            "• Blue containers/boxes\n"
            "• Yellow marks/indicators\n\n"
            "Features:\n"
            "• Define 9 custom detection boundaries\n"
            "• Real-time object detection\n"
            "• Multiple camera support\n"
            "• Custom YOLO model support\n"
            "• Visual status indicators\n"
            "• Relay control integration"
        )

    def closeEvent(self, event):
        """Handle application close event"""
        # Stop any running cameras
        if hasattr(self.training_page, 'camera') and self.training_page.camera:
            self.training_page.camera.release()
        if hasattr(self.detection_page, 'camera') and self.detection_page.camera:
            self.detection_page.camera.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()