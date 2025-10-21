self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_model_btn)
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        controls_layout.addWidget(self.start_btn)
        
        layout.addLayout(controls_layout)
        
        # Detection display
        self.detection_label = QLabel("Detection View")
        self.detection_label.setMinimumSize(800, 600)
        self.detection_label.setStyleSheet("border: 2px solid #333;")
        self.detection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detection_label)
        
        """
        Rewritten hikvision_3_v1.py

        This file is a cleaned, self-contained implementation of the original.
        It restores omitted sections, adds safe fallbacks for optional libraries
        and keeps a user-friendly PyQt UI for training and detection.

        Requirements: PyQt5, opencv-python, numpy. ultralytics and pyhid_usb_relay
        are optional — if missing the code will simulate behavior.
        """

        import sys
        import os
        import json
        import time
        import threading
        import queue
        import logging
        import traceback
        from datetime import datetime
        from collections import deque

        import cv2
        import numpy as np

        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *

        # Optional imports
        try:
            from ultralytics import YOLO
        except Exception:
            YOLO = None

        try:
            import pyhid_usb_relay
        except Exception:
            pyhid_usb_relay = None

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'detection_system_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)


        # FrameBuffer, WatchdogTimer, CameraThread, DetectionThread, RelayControlThread,
        # DrawingWidget, TrainingPage, DetectionPage, MainWindow implementations follow.

        class FrameBuffer:
            def __init__(self, maxsize=10):
                self.queue = queue.Queue(maxsize=maxsize)
                self.lock = threading.Lock()

            def put(self, frame, block=False):
                try:
                    self.queue.put(frame, block=block, timeout=0.001)
                except queue.Full:
                    try:
                        with self.lock:
                            try:
                                _ = self.queue.get_nowait()
                            except Exception:
                                pass
                        self.queue.put(frame, block=False)
                    except Exception:
                        pass

            def get(self, timeout=0.1):
                try:
                    return self.queue.get(timeout=timeout)
                except queue.Empty:
                    return None

            def clear(self):
                with self.lock:
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                        except Exception:
                            break


        class WatchdogTimer(QThread):
            timeout_signal = pyqtSignal(str)

            def __init__(self, name, timeout_seconds=10):
                super().__init__()
                self.name = name
                self.timeout_seconds = timeout_seconds
                self.last_heartbeat = time.time()
                self.running = True
                self.lock = threading.Lock()

            def heartbeat(self):
                with self.lock:
                    self.last_heartbeat = time.time()

            def run(self):
                logger.info(f"Watchdog started for {self.name}")
                while self.running:
                    time.sleep(1)
                    with self.lock:
                        elapsed = time.time() - self.last_heartbeat
                    if elapsed > self.timeout_seconds:
                        logger.critical(f"Watchdog timeout for {self.name} (elapsed={elapsed:.1f}s)")
                        try:
                            self.timeout_signal.emit(self.name)
                        except Exception:
                            pass
                        with self.lock:
                            self.last_heartbeat = time.time()

            def stop(self):
                self.running = False
                logger.info(f"Watchdog stopped for {self.name}")


        class CameraThread(QThread):
            frame_ready = pyqtSignal(object)
            error_signal = pyqtSignal(str)
            status_signal = pyqtSignal(str)

            def __init__(self, camera_source):
                super().__init__()
                self.camera_source = camera_source
                self.running = False
                self.camera = None
                self.frame_buffer = FrameBuffer(maxsize=5)
                self.reconnect_attempts = 0
                self.max_reconnect_attempts = 5
                self.frame_count = 0
                self.last_frame_time = time.time()
                self.is_ip_camera = isinstance(camera_source, str)

            def run(self):
                self.running = True
                camera_type = "IP Camera (RTSP)" if self.is_ip_camera else f"USB Camera {self.camera_source}"
                logger.info(f"Camera thread started for {camera_type}")

                while self.running:
                    try:
                        if self.camera is None or not getattr(self.camera, 'isOpened', lambda: False)():
                            if not self.connect_camera():
                                break

                        ret, frame = self.camera.read()
                        if not ret or frame is None:
                            logger.warning("Camera read failed, attempting reconnect")
                            self.reconnect_camera()
                            time.sleep(0.5)
                            continue

                        self.frame_buffer.put(frame.copy())
                        self.frame_ready.emit(frame)

                        self.frame_count += 1
                        now = time.time()
                        if now - self.last_frame_time >= 1.0:
                            fps = self.frame_count / (now - self.last_frame_time)
                            logger.debug(f"Camera FPS: {fps:.1f}")
                            self.frame_count = 0
                            self.last_frame_time = now

                        time.sleep(0.01)

                    except Exception as e:
                        logger.error(f"Camera thread error: {e}\n{traceback.format_exc()}")
                        try:
                            self.error_signal.emit(str(e))
                        except Exception:
                            pass
                        self.reconnect_camera()
                        time.sleep(1)

                self.cleanup()
                logger.info("Camera thread stopped")

            def connect_camera(self):
                for attempt in range(1, self.max_reconnect_attempts + 1):
                    try:
                        logger.info(f"Connecting to camera (attempt {attempt})")
                        if self.is_ip_camera:
                            self.camera = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                        else:
                            self.camera = cv2.VideoCapture(int(self.camera_source))

                        time.sleep(0.3)
                        if self.camera and self.camera.isOpened():
                            ret, test = self.camera.read()
                            if ret and test is not None:
                                self.reconnect_attempts = 0
                                self.status_signal.emit("Camera connected")
                                return True
                            else:
                                try:
                                    self.camera.release()
                                except Exception:
                                    pass
                        else:
                            logger.warning("Camera not opened yet")
                    except Exception as e:
                        logger.error(f"Error connecting camera: {e}")
                    time.sleep(1)

                try:
                    self.error_signal.emit(f"Failed to connect camera after {self.max_reconnect_attempts} attempts")
                except Exception:
                    pass
                return False

            def reconnect_camera(self):
                self.reconnect_attempts += 1
                try:
                    if self.camera:
                        self.camera.release()
                        self.camera = None
                except Exception:
                    pass

                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.status_signal.emit(f"Reconnecting... (attempt {self.reconnect_attempts})")
                    time.sleep(2)
                else:
                    self.error_signal.emit("Max reconnection attempts reached")
                    self.running = False

            def cleanup(self):
                try:
                    if self.camera:
                        self.camera.release()
                        self.camera = None
                except Exception:
                    pass
                logger.info("Camera resources cleaned up")

            def stop(self):
                self.running = False
                self.wait()


        class DetectionThread(QThread):
            detection_ready = pyqtSignal(dict)
            error_signal = pyqtSignal(str)
            fps_signal = pyqtSignal(float)

            def __init__(self, model_path, boundaries):
                super().__init__()
                self.model_path = model_path
                self.boundaries = boundaries
                self.blue_box_boundaries = [b for b in boundaries if b.get('type') == 'blue_box']
                self.running = False
                self.model = None
                self.frame_queue = queue.Queue(maxsize=5)
                self.fps_deque = deque(maxlen=30)
                self.last_detection_time = time.time()
                self.detection_count = 0

            def run(self):
                self.running = True
                logger.info("Detection thread started")
                try:
                    logger.info(f"Loading YOLO model: {self.model_path}")
                    if YOLO is not None:
                        try:
                            self.model = YOLO(self.model_path)
                            logger.info("YOLO model loaded successfully")
                        except Exception as e:
                            logger.warning(f"Failed to load YOLO model: {e} - falling back to color detection")
                            self.model = None
                    else:
                        logger.info("YOLO not available; using color-based fallback")
                        self.model = None

                    while self.running:
                        try:
                            frame = None
                            try:
                                frame = self.frame_queue.get(timeout=0.1)
                            except Exception:
                                time.sleep(0.01)
                                continue

                            if frame is None:
                                continue

                            start = time.time()
                            result = self.process_frame(frame)
                            elapsed = time.time() - start

                            self.detection_ready.emit(result)
                            self.detection_count += 1

                            self.fps_deque.append(elapsed)
                            if len(self.fps_deque) > 0:
                                avg = sum(self.fps_deque) / len(self.fps_deque)
                                fps = 1.0 / avg if avg > 0 else 0.0
                                self.fps_signal.emit(fps)

                        except Exception as e:
                            logger.error(f"Error in detection loop: {e}\n{traceback.format_exc()}")
                            try:
                                self.error_signal.emit(str(e))
                            except Exception:
                                pass
                            time.sleep(0.1)

                except Exception as e:
                    logger.critical(f"Fatal detection thread error: {e}\n{traceback.format_exc()}")
                    try:
                        self.error_signal.emit(f"Fatal error: {str(e)}")
                    except Exception:
                        pass

                logger.info("Detection thread stopped")

            def process_frame(self, frame):
                try:
                    blue_boxes = []
                    if self.model is not None:
                        results = self.model(frame, verbose=False, conf=0.5)
                        for res in results:
                            boxes = getattr(res, 'boxes', None)
                            if boxes is None:
                                continue
                            for box in boxes:
                                try:
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = map(float, xyxy)
                                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                                    cx = (x1 + x2) / 2.0
                                    cy = (y1 + y2) / 2.0
                                    blue_boxes.append({'bbox': (x1, y1, x2, y2), 'center': (cx, cy), 'confidence': conf})
                                except Exception:
                                    continue
                    else:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_blue = np.array([90, 60, 60])
                        upper_blue = np.array([140, 255, 255])
                        mask = cv2.inRange(hsv, lower_blue, upper_blue)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area < 200:
                                continue
                            x, y, w, h = cv2.boundingRect(cnt)
                            cx = x + w / 2
                            cy = y + h / 2
                            blue_boxes.append({'bbox': (x, y, x + w, y + h), 'center': (cx, cy), 'confidence': 0.9})

                    blue_status = []
                    for boundary in self.blue_box_boundaries:
                        objs = self.check_objects_in_boundary(blue_boxes, boundary)
                        blue_status.append(len(objs) > 0)

                    return {'frame': frame, 'blue_boxes': blue_boxes, 'blue_status': blue_status, 'timestamp': time.time()}
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    raise

            def check_objects_in_boundary(self, objects, boundary):
                objects_in_boundary = []
                for obj in objects:
                    center_x, center_y = obj['center']
                    if (boundary['x1'] <= center_x <= boundary['x2'] and boundary['y1'] <= center_y <= boundary['y2']):
                        objects_in_boundary.append(obj)
                return objects_in_boundary

            def add_frame(self, frame):
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass

            def stop(self):
                self.running = False
                self.wait()


        class RelayControlThread(QThread):
            error_signal = pyqtSignal(str)
            status_signal = pyqtSignal(str)

            def __init__(self):
                super().__init__()
                self.command_queue = queue.Queue()
                self.running = False
                self.relay = None
                self.last_state = None

            def run(self):
                self.running = True
                logger.info("Relay control thread started")
                while self.running:
                    try:
                        cmd = self.command_queue.get(timeout=0.5)
                        self.execute_command(cmd)
                        time.sleep(0.01)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Relay thread error: {e}\n{traceback.format_exc()}")
                        try:
                            self.error_signal.emit(str(e))
                        except Exception:
                            pass
                        time.sleep(0.5)
                logger.info("Relay control thread stopped")

            def execute_command(self, all_ok):
                try:
                    if all_ok == self.last_state:
                        return

                    if self.relay is None and pyhid_usb_relay is not None:
                        try:
                            self.relay = pyhid_usb_relay.find()
                        except Exception:
                            self.relay = None

                    if all_ok:
                        if self.relay:
                            try:
                                self.relay.set_state(1, True)
                                self.relay.set_state(2, False)
                                self.status_signal.emit("Relay: OK")
                            except Exception as e:
                                logger.error(f"Relay hardware error: {e}")
                                self.error_signal.emit(str(e))
                        else:
                            self.status_signal.emit("Relay: OK (simulated)")
                    else:
                        if self.relay:
                            try:
                                self.relay.set_state(1, False)
                                self.relay.set_state(2, True)
                                self.status_signal.emit("Relay: OFF")
                            except Exception as e:
                                logger.error(f"Relay hardware error: {e}")
                                self.error_signal.emit(str(e))
                        else:
                            self.status_signal.emit("Relay: OFF (simulated)")

                    self.last_state = all_ok
                except Exception as e:
                    logger.error(f"Relay execution error: {e}")
                    self.relay = None
                    try:
                        self.error_signal.emit(f"Relay error: {str(e)}")
                    except Exception:
                        pass

            def set_state(self, all_ok):
                try:
                    self.command_queue.put(all_ok, block=False)
                except queue.Full:
                    pass

            def stop(self):
                self.running = False
                self.wait()


        class DrawingWidget(QLabel):
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
                self.setMouseTracking(True)
                self.setCursor(Qt.CrossCursor)

                self.blue_box_color = QColor(0, 0, 255)
                self.preview_color = QColor(0, 255, 0, 100)
                self.line_width = 3

            def set_image(self, cv_image):
                self.cv_image = cv_image
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.original_pixmap = pixmap
                self.image = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                self.scale_x = pixmap.width() / self.image.width() if self.image.width() else 1.0
                self.scale_y = pixmap.height() / self.image.height() if self.image.height() else 1.0

                self.offset_x = (self.width() - self.image.width()) // 2
                self.offset_y = (self.height() - self.image.height()) // 2

                self.clear_boundaries()
                self.update_display()

            def resizeEvent(self, event):
                if hasattr(self, 'cv_image') and self.cv_image is not None:
                    self.set_image(self.cv_image)
                super().resizeEvent(event)

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton and self.image and len(self.rectangles) < 3:
                    img_pos = self.widget_to_image_coords(event.pos())
                    if img_pos:
                        self.drawing = True
                        self.start_point = img_pos

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

            def widget_to_image_coords(self, widget_point):
                if not self.image:
                    return None
                img_rect = QRect(self.offset_x, self.offset_y, self.image.width(), self.image.height())
                if not img_rect.contains(widget_point):
                    return None
                x = widget_point.x() - self.offset_x
                y = widget_point.y() - self.offset_y
                return QPoint(x, y)

            def finish_rectangle(self):
                if self.start_point and self.end_point:
                    x1 = min(self.start_point.x(), self.end_point.x())
                    y1 = min(self.start_point.y(), self.end_point.y())
                    x2 = max(self.start_point.x(), self.end_point.x())
                    y2 = max(self.start_point.y(), self.end_point.y())

                    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                        rect = QRect(x1, y1, x2 - x1, y2 - y1)
                        self.rectangles.append(rect)
                        boundary = {
                            'x1': x1 * self.scale_x,
                            'y1': y1 * self.scale_y,
                            'x2': x2 * self.scale_x,
                            'y2': y2 * self.scale_y,
                            'type': 'blue_box'
                        }
                        self.boundaries.append(boundary)
                        logger.info(f"Boundary added: blue_box - Total: {len(self.boundaries)}")
                        self.update_display()

            def update_display(self):
                if not self.image:
                    return
                display_pixmap = self.image.copy()
                painter = QPainter(display_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                for i, rect in enumerate(self.rectangles):
                    pen = QPen(self.blue_box_color, self.line_width)
                    painter.setPen(pen)
                    painter.drawRect(rect)
                    painter.setFont(QFont("Arial", 10, QFont.Bold))
                    label_pos = QPoint(rect.x() + 5, max(10, rect.y() - 5))
                    painter.drawText(label_pos, f"Blue Box {i + 1}")

                if self.drawing and self.start_point and self.end_point:
                    pen = QPen(self.preview_color, self.line_width)
                    painter.setPen(pen)
                    current_rect = QRect(min(self.start_point.x(), self.end_point.x()), min(self.start_point.y(), self.end_point.y()), abs(self.end_point.x() - self.start_point.x()), abs(self.end_point.y() - self.start_point.y()))
                    painter.drawRect(current_rect)
                painter.end()

                final_pixmap = QPixmap(self.size())
                final_pixmap.fill(Qt.white)
                final_painter = QPainter(final_pixmap)
                final_painter.drawPixmap(self.offset_x, self.offset_y, display_pixmap)
                final_painter.end()
                self.setPixmap(final_pixmap)

            def clear_boundaries(self):
                self.rectangles.clear()
                self.boundaries.clear()
                self.drawing = False
                if self.image:
                    self.update_display()
                logger.info("All boundaries cleared")

            def get_instruction_text(self):
                completed = len(self.rectangles)
                if completed < 3:
                    return f"Draw Blue Box boundary {completed + 1} of 3"
                else:
                    return "All 3 boundaries completed! Click 'Save Boundaries'"

                self.relay.set_state(2, False)
                logger.info("Relay: OK state (R1=ON, R2=OFF)")
                self.status_signal.emit("Relay: All OK")
            else:
                self.relay.set_state(1, False)
                self.relay.set_state(2, True)
                logger.warning("Relay: ERROR state (R1=OFF, R2=ON)")
                self.status_signal.emit("Relay: Issues Detected")
            
            self.last_state = all_ok
            
        except Exception as e:
            logger.error(f"Relay execution error: {e}")
            self.relay = None  # Reset relay connection
            self.error_signal.emit(f"Relay error: {str(e)}")
    
    def set_state(self, all_ok):
        """Queue relay state change"""
        try:
            self.command_queue.put(all_ok, block=False)
        except queue.Full:
            pass
    
    def stop(self):
        """Stop relay thread"""
        self.running = False
        self.wait()

class DrawingWidget(QLabel):
    """Custom widget for drawing boundaries"""
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
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        
        self.blue_box_color = QColor(0, 0, 255)
        self.yellow_mark_color = QColor(255, 255, 0)
        self.preview_color = QColor(0, 255, 0, 100)
        self.line_width = 3

    def set_image(self, cv_image):
        """Set the image for drawing boundaries"""
        self.cv_image = cv_image
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.original_pixmap = pixmap
        self.image = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.scale_x = pixmap.width() / self.image.width()
        self.scale_y = pixmap.height() / self.image.height()
        
        self.offset_x = (self.width() - self.image.width()) // 2
        self.offset_y = (self.height() - self.image.height()) // 2
        
        self.clear_boundaries()
        self.update_display()

    def resizeEvent(self, event):
        """Handle widget resizing"""
        if hasattr(self, 'cv_image') and self.cv_image is not None:
            self.set_image(self.cv_image)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image and len(self.rectangles) < 3:
            img_pos = self.widget_to_image_coords(event.pos())
            if img_pos:
                self.drawing = True
                self.start_point = img_pos

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

    def widget_to_image_coords(self, widget_point):
        """Convert widget coordinates to image coordinates"""
        if not self.image:
            return None
        
        img_rect = QRect(self.offset_x, self.offset_y, self.image.width(), self.image.height())
        if not img_rect.contains(widget_point):
            return None
        
        x = widget_point.x() - self.offset_x
        y = widget_point.y() - self.offset_y
        return QPoint(x, y)

    def finish_rectangle(self):
        """Finish drawing current rectangle"""
        if self.start_point and self.end_point:
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())
            
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                rect = QRect(x1, y1, x2 - x1, y2 - y1)
                self.rectangles.append(rect)
                
                boundary = {
                    'x1': x1 * self.scale_x,
                    'y1': y1 * self.scale_y,
                    'x2': x2 * self.scale_x,
                    'y2': y2 * self.scale_y,
                    'type': 'blue_box'
                }
                self.boundaries.append(boundary)
                logger.info(f"Boundary added: blue_box - Total: {len(self.boundaries)}")
                self.update_display()

    def update_display(self):
        """Update the display with current rectangles"""
        if not self.image:
            return
        
        display_pixmap = self.image.copy()
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for i, rect in enumerate(self.rectangles):
            color = self.blue_box_color
            label = f"Blue Box {i + 1}"
            
            pen = QPen(color, self.line_width)
            painter.setPen(pen)
            painter.drawRect(rect)
            
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            label_pos = QPoint(rect.x() + 5, rect.y() - 5)
            painter.drawText(label_pos, label)
        
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
        logger.info("All boundaries cleared")

    def get_instruction_text(self):
        """Get current instruction text"""
        completed = len(self.rectangles)
        if completed < 3:
            return f"Draw Blue Box boundary {completed + 1} of 3"
        else:
            return "All 3 boundaries completed! Click 'Save Boundaries'"

class TrainingPage(QWidget):
    """Training page with camera control - supports USB and IP cameras"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera_thread = None
        self.ip_camera_url = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Training - Define Container Boundaries")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        instructions = QLabel(
            "Instructions:\n"
            "Draw 3 boundaries for BLUE BOXES\n"
            "Total: 3 boundaries required"
        )
        instructions.setStyleSheet(
            "background-color: #E1F5FE; padding: 10px; border-radius: 5px; "
            "font-size: 12px; color: #0277BD; border: 1px solid #0288D1;"
        )
        layout.addWidget(instructions)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        # Camera type selection
        camera_type_layout = QVBoxLayout()
        self.camera_type_group = QButtonGroup()
        self.usb_radio = QRadioButton("USB Camera")
        self.ip_radio = QRadioButton("IP Camera (RTSP)")
        self.usb_radio.setChecked(True)
        self.camera_type_group.addButton(self.usb_radio)
        self.camera_type_group.addButton(self.ip_radio)
        camera_type_layout.addWidget(self.usb_radio)
        camera_type_layout.addWidget(self.ip_radio)
        camera_controls.addLayout(camera_type_layout)
        
        # USB camera selection
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        camera_controls.addWidget(QLabel("USB:"))
        camera_controls.addWidget(self.camera_combo)
        
        # IP camera URL input
        camera_controls.addWidget(QLabel("RTSP URL:"))
        self.ip_url_input = QLineEdit()
        self.ip_url_input.setPlaceholderText("rtsp://admin:password@192.168.1.10:554/stream")
        self.ip_url_input.setText("rtsp://admin:lampsun_123@192.168.1.10:554/stream")
        self.ip_url_input.setMinimumWidth(350)
        camera_controls.addWidget(self.ip_url_input)
        
        # Radio button toggle
        self.usb_radio.toggled.connect(self.on_camera_type_changed)
        self.ip_radio.toggled.connect(self.on_camera_type_changed)
        
        layout.addLayout(camera_controls)
        
        # Camera control buttons
        button_controls = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        button_controls.addWidget(self.start_camera_btn)
        
        self.capture_btn = QPushButton("📷 Capture Photo")
        self.capture_btn.clicked.connect(self.capture_frame)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        button_controls.addWidget(self.capture_btn)
        
        layout.addLayout(button_controls)
        
        self.image_area = QStackedWidget()
        
        self.camera_label = QLabel("📹 Start camera to see preview")
        self.camera_label.setMinimumSize(900, 600)
        self.camera_label.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5; font-size: 16px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.image_area.addWidget(self.camera_label)
        
        self.drawing_widget = DrawingWidget()
        self.drawing_widget.setMinimumSize(900, 600)
        self.image_area.addWidget(self.drawing_widget)
        
        layout.addWidget(self.image_area)
        
        self.instruction_label = QLabel("📷 Click 'Capture Photo' to start")
        self.instruction_label.setStyleSheet(
            "background-color: #E3F2FD; padding: 10px; border-radius: 5px; "
            "font-size: 12px; color: #1976D2;"
        )
        layout.addWidget(self.instruction_label)
        
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_boundaries)
        self.clear_btn.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
        button_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("💾 Save Boundaries")
        self.save_btn.clicked.connect(self.save_boundaries)
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        self.status_label = QLabel("✅ Ready")
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: #E8F5E8; color: #2E7D32; "
            "border: 1px solid #4CAF50; border-radius: 5px; font-weight: bold;"
        )
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Initial state
        self.on_camera_type_changed()

    def on_camera_type_changed(self):
        """Handle camera type selection change"""
        is_usb = self.usb_radio.isChecked()
        self.camera_combo.setEnabled(is_usb)
        self.ip_url_input.setEnabled(not is_usb)

    def refresh_cameras(self):
        """Refresh available USB cameras"""
        self.camera_combo.clear()
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()

    def start_camera(self):
        """Start camera with thread"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        else:
            # Determine camera source
            if self.usb_radio.isChecked():
                camera_source = self.camera_combo.currentIndex()
            else:
                camera_source = self.ip_url_input.text().strip()
                if not camera_source:
                    QMessageBox.warning(self, "Error", "Please enter RTSP URL!")
                    return
                if not camera_source.startswith("rtsp://"):
                    QMessageBox.warning(self, "Error", "Invalid RTSP URL! Must start with rtsp://")
                    return
            
            self.camera_thread = CameraThread(camera_source)
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.error_signal.connect(self.handle_camera_error)
            self.camera_thread.status_signal.connect(self.update_status)
            self.camera_thread.start()
            
            self.start_camera_btn.setText("⏹️ Stop Camera")
            self.start_camera_btn.setStyleSheet("background-color: #F44336; color: white; padding: 8px;")
            self.capture_btn.setEnabled(True)
            self.image_area.setCurrentWidget(self.camera_label)

    def stop_camera(self):
        """Stop camera thread"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.camera_label.clear()
        self.camera_label.setText("📹 Start camera to see preview")
        self.start_camera_btn.setText("Start Camera")
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.capture_btn.setEnabled(False)
        self.image_area.setCurrentWidget(self.camera_label)

    def update_frame(self, frame):
        """Update camera preview"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
        self.current_frame = frame.copy()

    def capture_frame(self):
        """Capture frame for training"""
        if hasattr(self, 'current_frame'):
            self.stop_camera()
            self.drawing_widget.set_image(self.current_frame)
            self.captured_frame = self.current_frame.copy()
            self.image_area.setCurrentWidget(self.drawing_widget)
            self.update_instruction_text()
            logger.info("Frame captured for training")
        else:
            QMessageBox.warning(self, "Error", "No frame available!")

    def update_instruction_text(self):
        """Update instruction text"""
        instruction_text = self.drawing_widget.get_instruction_text()
        self.instruction_label.setText(f"✏️ {instruction_text}")

    def clear_boundaries(self):
        """Clear all boundaries"""
        self.drawing_widget.clear_boundaries()
        self.update_instruction_text()

    def save_boundaries(self):
        """Save boundaries to file"""
        if len(self.drawing_widget.boundaries) != 3:
            QMessageBox.warning(self, "Error", "Please draw exactly 3 boundaries for blue boxes!")
            return
        
        blue_box_boundaries = [b for b in self.drawing_widget.boundaries if b.get('type') == 'blue_box']
        
        data = {
            'blue_box_boundaries': blue_box_boundaries,
            'all_boundaries': self.drawing_widget.boundaries,
            'frame_shape': self.captured_frame.shape
        }
        
        try:
            with open('boundaries.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            cv2.imwrite('reference_frame.jpg', self.captured_frame)
            
            logger.info(f"Boundaries saved: {len(blue_box_boundaries)} blue boxes")
            
            QMessageBox.information(
                self, "Success",
                f"✅ Boundaries saved successfully!\n"
                f"🔵 Blue boxes: {len(blue_box_boundaries)}"
            )
        except Exception as e:
            logger.error(f"Failed to save boundaries: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def handle_camera_error(self, error):
        """Handle camera errors"""
        logger.error(f"Camera error: {error}")
        QMessageBox.warning(self, "Camera Error", error)

    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(f"📹 {status}")

    def closeEvent(self, event):
        """Cleanup on close"""
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()

class DetectionPage(QWidget):
    """Industrial-grade detection page with multi-threading - supports USB and IP cameras"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera_thread = None
        self.detection_thread = None
        self.relay_thread = None
        self.watchdog = None
        self.boundaries = []
        self.blue_box_boundaries = []
        self.model_path = None
        self.running = False
        self.detection_count = 0
        self.error_count = 0
        self.last_relay_state = None
        self.detection_history = deque(maxlen=100)
        self.uptime_start = None
        self.ip_camera_url = None
        self.load_boundaries()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Detection - Industrial Monitoring System")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Camera type selection
        camera_type_layout = QVBoxLayout()
        self.camera_type_group = QButtonGroup()
        self.usb_radio = QRadioButton("USB")
        self.ip_radio = QRadioButton("IP Camera")
        self.usb_radio.setChecked(True)
        self.camera_type_group.addButton(self.usb_radio)
        self.camera_type_group.addButton(self.ip_radio)
        camera_type_layout.addWidget(self.usb_radio)
        camera_type_layout.addWidget(self.ip_radio)
        controls_layout.addLayout(camera_type_layout)
        
        # USB camera selection
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        controls_layout.addWidget(QLabel("USB:"))
        controls_layout.addWidget(self.camera_combo)
        
        # IP camera URL input
        controls_layout.addWidget(QLabel("RTSP:"))
        self.ip_url_input = QLineEdit()
        self.ip_url_input.setPlaceholderText("rtsp://admin:pass@192.168.1.10:554/stream")
        self.ip_url_input.setText("rtsp://admin:lampsun_123@192.168.1.10:554/stream")
        self.ip_url_input.setMinimumWidth(300)
        controls_layout.addWidget(self.ip_url_input)
        
        # Radio button toggle
        self.usb_radio.toggled.connect(self.on_camera_type_changed)
        self.ip_radio.toggled.connect(self.on_camera_type_changed)