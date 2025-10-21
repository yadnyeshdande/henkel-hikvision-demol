import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QMessageBox, QComboBox, QFileDialog, QFrame,
                             QSizePolicy, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import os
import json

# Try to import YOLO
YOLO_AVAILABLE = False
try:
    # Import may raise ImportError or OSError (DLL init failure). Catch all to avoid crashing.
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
    print("✓ YOLO module loaded successfully")
except Exception as e:
    # Provide a helpful message about why import failed without stopping the app.
    print("⚠ YOLO module not available or failed to initialize.")
    print("  Reason:", repr(e))
    print("  Install ultralytics or fix PyTorch installation: pip install ultralytics")

# Try to import USB relay
RELAY_AVAILABLE = False
pyhid_usb_relay = None
try:
    import pyhid_usb_relay  # type: ignore
    RELAY_AVAILABLE = True
    print("✓ pyhid_usb_relay module loaded successfully")
except Exception as e:
    print("⚠ USB Relay module not available or failed to initialize.")
    print("  Reason:", repr(e))
    print("  Install pyhid USB relay package: pip install pyhid-usb-relay")


class VideoWidget(QLabel):
    """Custom widget for displaying video with shape drawing"""
    
    def __init__(self, enable_drawing=False):
        super().__init__()
        self.enable_drawing = enable_drawing
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #444;")
        self.setAlignment(Qt.AlignCenter)
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.rectangles = []
        self.current_frame = None
        self.frame_size = None
        
    def set_frame(self, frame):
        """Set the current video frame"""
        self.current_frame = frame.copy()
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.update_display()
        
    def update_display(self, overlay_rect=None):
        """Update the display with current frame and shapes"""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # Draw saved rectangles
        for rect in self.rectangles:
            x1, y1, x2, y2 = rect
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            
            # Add corner handles
            handle_size = 8
            cv2.circle(display_frame, (x1, y1), handle_size, (255, 255, 0), -1)
            cv2.circle(display_frame, (x2, y1), handle_size, (255, 255, 0), -1)
            cv2.circle(display_frame, (x1, y2), handle_size, (255, 255, 0), -1)
            cv2.circle(display_frame, (x2, y2), handle_size, (255, 255, 0), -1)
        
        # Draw current rectangle being drawn
        if overlay_rect:
            x1, y1, x2, y2 = overlay_rect
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)
        
        # Convert to QPixmap
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
    
    def widget_to_frame_coords(self, widget_point):
        """Convert widget coordinates to frame coordinates"""
        if not self.pixmap() or not self.frame_size:
            return None
        
        # Get the actual displayed pixmap size
        pixmap_size = self.pixmap().size()
        widget_size = self.size()
        
        # Calculate offset (pixmap is centered)
        x_offset = (widget_size.width() - pixmap_size.width()) // 2
        y_offset = (widget_size.height() - pixmap_size.height()) // 2
        
        # Adjust for offset
        adjusted_x = widget_point.x() - x_offset
        adjusted_y = widget_point.y() - y_offset
        
        # Check if click is within pixmap bounds
        if adjusted_x < 0 or adjusted_y < 0 or adjusted_x >= pixmap_size.width() or adjusted_y >= pixmap_size.height():
            return None
        
        # Scale to frame coordinates
        scale_x = self.frame_size[0] / pixmap_size.width()
        scale_y = self.frame_size[1] / pixmap_size.height()
        
        frame_x = int(adjusted_x * scale_x)
        frame_y = int(adjusted_y * scale_y)
        
        return (frame_x, frame_y)
    
    def mousePressEvent(self, event):
        if not self.enable_drawing or event.button() != Qt.LeftButton:
            return
        
        coords = self.widget_to_frame_coords(event.pos())
        if coords:
            self.drawing = True
            self.start_point = coords
            self.current_point = coords
    
    def mouseMoveEvent(self, event):
        if not self.enable_drawing or not self.drawing:
            return
        
        coords = self.widget_to_frame_coords(event.pos())
        if coords:
            self.current_point = coords
            x1, y1 = self.start_point
            x2, y2 = self.current_point
            self.update_display(overlay_rect=(x1, y1, x2, y2))
    
    def mouseReleaseEvent(self, event):
        if not self.enable_drawing or event.button() != Qt.LeftButton or not self.drawing:
            return
        
        coords = self.widget_to_frame_coords(event.pos())
        if coords and self.start_point:
            x1, y1 = self.start_point
            x2, y2 = coords
            
            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Only add if rectangle has meaningful size
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.rectangles.append((x1, y1, x2, y2))
        
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.update_display()
    
    def clear_rectangles(self):
        """Clear all drawn rectangles"""
        self.rectangles = []
        self.update_display()
    
    def get_rectangles(self):
        """Get all drawn rectangles"""
        return self.rectangles


class NavigationBar(QFrame):
    """Top navigation bar for page switching"""
    teachingClicked = pyqtSignal()
    detectionClicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-bottom: 3px solid #3498db;
            }
        """)
        self.setFixedHeight(80)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Logo/Title
        title = QLabel("⚙ POKAYOKE VISION SYSTEM")
        title.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Navigation buttons
        self.teaching_btn = QPushButton("📐 TEACHING MODE")
        self.teaching_btn.setStyleSheet(self.get_nav_button_style(True))
        self.teaching_btn.setFixedSize(200, 50)
        self.teaching_btn.clicked.connect(self.teachingClicked.emit)
        layout.addWidget(self.teaching_btn)
        
        self.detection_btn = QPushButton("🎯 DETECTION MODE")
        self.detection_btn.setStyleSheet(self.get_nav_button_style(False))
        self.detection_btn.setFixedSize(200, 50)
        self.detection_btn.clicked.connect(self.detectionClicked.emit)
        layout.addWidget(self.detection_btn)
        
        self.setLayout(layout)
    
    def get_nav_button_style(self, active=False):
        if active:
            return """
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #34495e;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3498db;
                }
            """
    
    def set_active_page(self, page):
        if page == "teaching":
            self.teaching_btn.setStyleSheet(self.get_nav_button_style(True))
            self.detection_btn.setStyleSheet(self.get_nav_button_style(False))
        else:
            self.teaching_btn.setStyleSheet(self.get_nav_button_style(False))
            self.detection_btn.setStyleSheet(self.get_nav_button_style(True))


class TeachingPage(QWidget):
    """Page for teaching/defining restricted areas"""
    def __init__(self, app_controller):
        super().__init__()
        self.app_controller = app_controller
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Control Panel
        control_panel = QGroupBox("Configuration")
        control_panel.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        control_layout = QHBoxLayout()
        
        # Camera selection
        control_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2", "Camera 3"])
        self.camera_combo.setFixedWidth(120)
        control_layout.addWidget(self.camera_combo)
        
        control_layout.addSpacing(20)
        
        # Camera controls
        self.start_camera_btn = QPushButton("▶ Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setStyleSheet(self.get_button_style("#27ae60"))
        self.start_camera_btn.setFixedHeight(40)
        control_layout.addWidget(self.start_camera_btn)
        
        self.stop_camera_btn = QPushButton("⏸ Stop Camera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        self.stop_camera_btn.setStyleSheet(self.get_button_style("#e74c3c"))
        self.stop_camera_btn.setFixedHeight(40)
        control_layout.addWidget(self.stop_camera_btn)
        
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # Content layout (video + side panel)
        content_layout = QHBoxLayout()
        
        # Video display
        self.video_widget = VideoWidget(enable_drawing=True)
        content_layout.addWidget(self.video_widget, stretch=3)
        
        # Side panel
        side_panel = QFrame()
        side_panel.setFrameStyle(QFrame.StyledPanel)
        side_panel.setStyleSheet("background-color: #ecf0f1; border-radius: 5px;")
        side_panel.setFixedWidth(280)
        side_layout = QVBoxLayout()
        side_layout.setSpacing(15)
        
        # Instructions
        instructions = QLabel(
            "<b>How to Use:</b><br><br>"
            "1. Select camera and click 'Start Camera'<br><br>"
            "2. <b>Draw restricted areas:</b><br>"
            "   • Click and drag to draw rectangles<br>"
            "   • Draw multiple areas if needed<br><br>"
            "3. Click 'Clear All' to restart<br><br>"
            "4. Click 'Save Configuration' when done"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 15px; font-size: 12px; background-color: white; border-radius: 5px;")
        side_layout.addWidget(instructions)
        
        # Area management
        self.clear_btn = QPushButton("🗑 Clear All Areas")
        self.clear_btn.clicked.connect(self.clear_areas)
        self.clear_btn.setStyleSheet(self.get_button_style("#95a5a6"))
        side_layout.addWidget(self.clear_btn)
        
        self.areas_label = QLabel("Restricted Areas: 0")
        self.areas_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 10px; background-color: white; border-radius: 5px;")
        self.areas_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(self.areas_label)
        
        side_layout.addStretch()
        
        # Save button
        self.save_btn = QPushButton("💾 Save Configuration")
        self.save_btn.clicked.connect(self.save_configuration)
        self.save_btn.setStyleSheet(self.get_button_style("#3498db", 50))
        side_layout.addWidget(self.save_btn)
        
        side_panel.setLayout(side_layout)
        content_layout.addWidget(side_panel)
        
        main_layout.addLayout(content_layout, stretch=1)
        
        self.setLayout(main_layout)
        
        # Update area count timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_area_count)
        self.update_timer.start(100)
    
    def get_button_style(self, color, height=40):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                min-height: {height}px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #7f8c8d;
            }}
        """
    
    def start_camera(self):
        camera_index = self.camera_combo.currentIndex()
        self.camera = cv2.VideoCapture(camera_index)
        
        if self.camera.isOpened():
            self.timer.start(30)
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
        else:
            QMessageBox.critical(self, "Camera Error", "Could not open the selected camera!")
    
    def stop_camera(self):
        self.timer.stop()
        if self.camera:
            self.camera.release()
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
    
    def update_frame(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.video_widget.set_frame(frame)
    
    def clear_areas(self):
        reply = QMessageBox.question(self, 'Confirm Clear', 
                                     'Are you sure you want to clear all restricted areas?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.video_widget.clear_rectangles()
    
    def update_area_count(self):
        count = len(self.video_widget.get_rectangles())
        self.areas_label.setText(f"Restricted Areas: {count}")
    
    def save_configuration(self):
        rectangles = self.video_widget.get_rectangles()
        if not rectangles:
            QMessageBox.warning(self, "No Areas Defined", 
                              "Please draw at least one restricted area before saving!")
            return
        
        camera_index = self.camera_combo.currentIndex()
        config = {
            'restricted_areas': rectangles,
            'camera_index': camera_index
        }
        
        try:
            with open('pokayoke_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            QMessageBox.information(self, "Configuration Saved", 
                                  f"Successfully saved {len(rectangles)} restricted area(s)!\n\n"
                                  "You can now switch to Detection Mode.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration:\n{str(e)}")


class DetectionPage(QWidget):
    """Page for real-time detection and monitoring"""
    def __init__(self, app_controller):
        super().__init__()
        self.app_controller = app_controller
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_detection)
        self.restricted_areas = []
        self.camera_index = 0
        self.relay_active = False
        self.model = None
        self.model_path = None
        
        # USB Relay
        self.relay = None
        if RELAY_AVAILABLE:
            try:
                # Find the connected USB relay board using pyhid_usb_relay API
                self.relay = pyhid_usb_relay.find()
                if not self.relay:
                    raise RuntimeError("No USB relay board found")
                print("✓ USB Relay initialized")
            except Exception as e:
                print(f"⚠ Relay initialization failed: {e}")
                self.relay = None
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Control Panel
        control_panel = QGroupBox("Detection Configuration")
        control_panel.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #e74c3c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        control_layout = QHBoxLayout()
        
        # Model selection
        control_layout.addWidget(QLabel("YOLO Model:"))
        self.model_label = QLabel("No model loaded" if not YOLO_AVAILABLE else "Ready to load model")
        self.model_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        control_layout.addWidget(self.model_label)
        
        self.load_model_btn = QPushButton("📁 Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setEnabled(YOLO_AVAILABLE)
        self.load_model_btn.setStyleSheet(self.get_button_style("#9b59b6"))
        self.load_model_btn.setFixedHeight(40)
        if not YOLO_AVAILABLE:
            self.load_model_btn.setToolTip("Install ultralytics: pip install ultralytics")
        control_layout.addWidget(self.load_model_btn)
        
        control_layout.addSpacing(20)
        
        # Detection controls
        self.start_detection_btn = QPushButton("▶ Start Detection")
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.start_detection_btn.setStyleSheet(self.get_button_style("#27ae60"))
        self.start_detection_btn.setFixedHeight(40)
        control_layout.addWidget(self.start_detection_btn)
        
        self.stop_detection_btn = QPushButton("⏹ Stop Detection")
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        self.stop_detection_btn.setEnabled(False)
        self.stop_detection_btn.setStyleSheet(self.get_button_style("#e74c3c"))
        self.stop_detection_btn.setFixedHeight(40)
        control_layout.addWidget(self.stop_detection_btn)
        
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # Status message for missing dependencies
        if not YOLO_AVAILABLE:
            warning_frame = QFrame()
            warning_frame.setStyleSheet("background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 5px; padding: 10px;")
            warning_layout = QHBoxLayout()
            warning_label = QLabel("⚠ <b>YOLO not installed.</b> Install with: <code>pip install ultralytics</code>")
            warning_label.setStyleSheet("color: #856404; font-size: 12px;")
            warning_layout.addWidget(warning_label)
            warning_frame.setLayout(warning_layout)
            main_layout.addWidget(warning_frame)
        
        # Content layout
        content_layout = QHBoxLayout()
        
        # Video display
        self.video_widget = VideoWidget(enable_drawing=False)
        content_layout.addWidget(self.video_widget, stretch=3)
        
        # Side panel
        side_panel = QFrame()
        side_panel.setFrameStyle(QFrame.StyledPanel)
        side_panel.setStyleSheet("background-color: #ecf0f1; border-radius: 5px;")
        side_panel.setFixedWidth(280)
        side_layout = QVBoxLayout()
        side_layout.setSpacing(15)
        
        # Status display
        status_group = QGroupBox("System Status")
        status_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("● INACTIVE")
        self.status_label.setStyleSheet("font-size: 16px; color: #95a5a6; font-weight: bold; padding: 10px; background-color: white; border-radius: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        self.alarm_status = QLabel("🔔 ALARM: OFF")
        self.alarm_status.setStyleSheet("font-size: 16px; color: white; font-weight: bold; padding: 15px; background-color: #27ae60; border-radius: 5px;")
        self.alarm_status.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.alarm_status)
        
        self.detection_count = QLabel("Detections: 0")
        self.detection_count.setStyleSheet("font-size: 13px; padding: 10px; background-color: white; border-radius: 5px;")
        self.detection_count.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.detection_count)
        
        status_group.setLayout(status_layout)
        side_layout.addWidget(status_group)
        
        # Legend
        legend_group = QGroupBox("Detection Legend")
        legend_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        legend_layout = QVBoxLayout()
        
        safe_label = QLabel("🟢 Green Box = Safe Area")
        safe_label.setStyleSheet("padding: 5px; font-size: 12px;")
        legend_layout.addWidget(safe_label)
        
        danger_label = QLabel("🔴 Red Box = RESTRICTED!")
        danger_label.setStyleSheet("padding: 5px; font-size: 12px; font-weight: bold;")
        legend_layout.addWidget(danger_label)
        
        legend_group.setLayout(legend_layout)
        side_layout.addWidget(legend_group)
        
        side_layout.addStretch()
        
        side_panel.setLayout(side_layout)
        content_layout.addWidget(side_panel)
        
        main_layout.addLayout(content_layout, stretch=1)
        
        self.setLayout(main_layout)
        
        self.detection_counter = 0
    
    def get_button_style(self, color, height=40):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                min-height: {height}px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #7f8c8d;
            }}
        """
    
    def load_model(self):
        if not YOLO_AVAILABLE:
            QMessageBox.warning(
                self,
                "YOLO Not Available",
                "YOLO is not installed.\n\n"
                "Please install it using:\n"
                "pip install ultralytics\n\n"
                "Then restart the application."
            )
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "YOLO Models (*.pt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_path = file_path
                model_name = os.path.basename(file_path)
                self.model_label.setText(f"✓ {model_name}")
                self.model_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                QMessageBox.information(self, "Model Loaded", 
                                      f"Successfully loaded model:\n{model_name}")
            except Exception as e:
                QMessageBox.critical(self, "Model Error", 
                                   f"Failed to load model:\n{str(e)}")
    
    def load_config(self):
        """Load restricted area configuration"""
        if not os.path.exists('pokayoke_config.json'):
            return False
        
        try:
            with open('pokayoke_config.json', 'r') as f:
                config = json.load(f)
                self.restricted_areas = config['restricted_areas']
                self.camera_index = config['camera_index']
                return True
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to load configuration:\n{str(e)}")
            return False
    
    def start_detection(self):
        if not self.load_config():
            QMessageBox.warning(self, "Configuration Missing", 
                              "No configuration found!\n\nPlease go to Teaching Mode and define restricted areas first.")
            return
        
        if not YOLO_AVAILABLE:
            QMessageBox.warning(self, "YOLO Not Available", 
                              "YOLO is not installed!\n\n"
                              "Please install it using:\n"
                              "pip install ultralytics\n\n"
                              "Then restart the application.")
            return
        
        if not self.model:
            QMessageBox.warning(self, "Model Missing", 
                              "No YOLO model loaded!\n\nPlease load a YOLO model before starting detection.")
            return
        
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if self.camera.isOpened():
            self.timer.start(30)
            self.start_detection_btn.setEnabled(False)
            self.stop_detection_btn.setEnabled(True)
            self.status_label.setText("● ACTIVE")
            self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; font-weight: bold; padding: 10px; background-color: white; border-radius: 5px;")
        else:
            QMessageBox.critical(self, "Camera Error", "Could not open camera!")
    
    def stop_detection(self):
        self.timer.stop()
        if self.camera:
            self.camera.release()
        self.start_detection_btn.setEnabled(True)
        self.stop_detection_btn.setEnabled(False)
        self.status_label.setText("● INACTIVE")
        self.status_label.setStyleSheet("font-size: 16px; color: #95a5a6; font-weight: bold; padding: 10px; background-color: white; border-radius: 5px;")
        self.turn_off_relay()
    
    def update_detection(self):
        if not self.camera or not self.camera.isOpened():
            return
        
        ret, frame = self.camera.read()
        if not ret:
            return
        
        person_in_restricted = False
        
        # Draw restricted areas
        for area in self.restricted_areas:
            x1, y1, x2, y2 = area
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.putText(frame, "RESTRICTED", (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # YOLO Detection
        if YOLO_AVAILABLE and self.model:
            try:
                results = self.model(frame, classes=[0], verbose=False)  # class 0 is 'person'
                self.detection_counter = 0
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Calculate center point of bounding box
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Check if person is in any restricted area
                        in_restricted = False
                        for area in self.restricted_areas:
                            ax1, ay1, ax2, ay2 = area
                            if ax1 <= center_x <= ax2 and ay1 <= center_y <= ay2:
                                in_restricted = True
                                person_in_restricted = True
                                break
                            if ax1 <= center_x <= ax2 and ay1 <= center_y <= ay2:
                                in_restricted = True
                                break
                        
                        # Draw bounding box
                        color = (0, 0, 255) if in_restricted else (0, 255, 0)
                        thickness = 3 if in_restricted else 2
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                        cv2.circle(frame, (center_x, center_y), 5, color, -1)
                        
                        # Label
                        label = f"Person {conf:.2f}"
                        if in_restricted:
                            label = f"⚠ ALERT! {conf:.2f}"
                            person_in_restricted = True
                            # Draw flashing background
                            cv2.rectangle(frame, (int(x1), int(y1) - 35), 
                                        (int(x1) + 200, int(y1)), (0, 0, 255), -1)
                        
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        self.detection_counter += 1
                
                self.detection_count.setText(f"Detections: {self.detection_counter}")
            
            except Exception as e:
                print(f"Detection error: {e}")
                cv2.putText(frame, f"Detection Error: {str(e)[:50]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Show message if YOLO is not available
            cv2.putText(frame, "YOLO NOT AVAILABLE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Install: pip install ultralytics", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Control relay
        if person_in_restricted:
            self.turn_on_relay()
        else:
            self.turn_off_relay()
        
        # Display frame
        self.video_widget.set_frame(frame)
    
    def turn_on_relay(self):
        """Turn on USB relay for alarm"""
        if not self.relay_active:
            self.relay_active = True
            self.alarm_status.setText("🔔 ALARM: ON")
            self.alarm_status.setStyleSheet("font-size: 16px; color: white; font-weight: bold; padding: 15px; background-color: #e74c3c; border-radius: 5px;")
            
            # Activate relay
            if RELAY_AVAILABLE and self.relay:
                try:
                    # set_state(channel, True) to turn on channel 1
                    self.relay.set_state(1, True)
                    print("✓ Relay activated")
                except Exception as e:
                    print(f"⚠ Relay error: {e}")
            
            print("⚠ ALARM ACTIVATED - Person in restricted area!")
    
    def turn_off_relay(self):
        """Turn off USB relay"""
        if self.relay_active:
            self.relay_active = False
            self.alarm_status.setText("🔔 ALARM: OFF")
            self.alarm_status.setStyleSheet("font-size: 16px; color: white; font-weight: bold; padding: 15px; background-color: #27ae60; border-radius: 5px;")
            
            # Deactivate relay
            if RELAY_AVAILABLE and self.relay:
                try:
                    # set_state(channel, False) to turn off channel 1
                    self.relay.set_state(1, False)
                    print("✓ Relay deactivated")
                except Exception as e:
                    print(f"⚠ Relay error: {e}")
            
            print("✓ Alarm deactivated - Area clear")


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pokayoke Vision Safety System - Industrial Grade")
        
        # Set window to maximize but not fullscreen
        screen = QApplication.desktop().screenGeometry()
        self.setGeometry(50, 50, screen.width() - 100, screen.height() - 100)
        
        # Set minimum size
        self.setMinimumSize(1280, 720)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Navigation bar
        self.nav_bar = NavigationBar()
        self.nav_bar.teachingClicked.connect(self.switch_to_teaching)
        self.nav_bar.detectionClicked.connect(self.switch_to_detection)
        main_layout.addWidget(self.nav_bar)
        
        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.teaching_page = TeachingPage(self)
        self.detection_page = DetectionPage(self)
        
        self.stacked_widget.addWidget(self.teaching_page)
        self.stacked_widget.addWidget(self.detection_page)
        
        # Start with teaching page
        self.stacked_widget.setCurrentIndex(0)
        self.nav_bar.set_active_page("teaching")
        
        main_widget.setLayout(main_layout)
        
        # Apply global stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
            }
        """)
        
        # Show startup message
        self.show_startup_info()
    
    def show_startup_info(self):
        """Show information about available features"""
        msg = "Pokayoke Vision System\n\n"
        msg += "System Status:\n"
        
        if YOLO_AVAILABLE:
            msg += "✓ YOLO Detection: Available\n"
        else:
            msg += "✗ YOLO Detection: Not installed\n"
            msg += "  Install: pip install ultralytics\n\n"
        
        if RELAY_AVAILABLE:
            msg += "✓ USB Relay: Available\n"
        else:
            msg += "✗ USB Relay: Not installed\n"
            msg += "  Install: pip install pyhid-usb-relay\n\n"
        
        if not YOLO_AVAILABLE or not RELAY_AVAILABLE:
            msg += "\nThe system will work in limited mode.\n"
            msg += "Install missing components for full functionality."
            
            QMessageBox.information(self, "System Status", msg)
    
    def switch_to_teaching(self):
        # Stop detection if running
        if hasattr(self.detection_page, 'timer') and self.detection_page.timer.isActive():
            reply = QMessageBox.question(
                self, 
                'Switch Page', 
                'Detection is currently active. Do you want to stop it and switch to Teaching Mode?',
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.detection_page.stop_detection()
            else:
                return
        
        self.stacked_widget.setCurrentIndex(0)
        self.nav_bar.set_active_page("teaching")
    
    def switch_to_detection(self):
        # Stop camera in teaching if running
        if hasattr(self.teaching_page, 'timer') and self.teaching_page.timer.isActive():
            self.teaching_page.stop_camera()
        
        self.stacked_widget.setCurrentIndex(1)
        self.nav_bar.set_active_page("detection")
    
    def closeEvent(self, event):
        """Clean up resources on close"""
        reply = QMessageBox.question(
            self,
            'Exit Application',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Stop all cameras
            if hasattr(self.teaching_page, 'camera') and self.teaching_page.camera:
                self.teaching_page.camera.release()
            if hasattr(self.detection_page, 'camera') and self.detection_page.camera:
                self.detection_page.camera.release()
                self.detection_page.turn_off_relay()
            
            cv2.destroyAllWindows()
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Print system info
    print("\n" + "="*60)
    print("POKAYOKE VISION SAFETY SYSTEM")
    print("="*60)
    print(f"YOLO Available: {YOLO_AVAILABLE}")
    print(f"USB Relay Available: {RELAY_AVAILABLE}")
    print("="*60 + "\n")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()