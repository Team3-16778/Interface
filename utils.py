import sys
import re
import serial
from serial.tools import list_ports
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QLCDNumber, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont, QDoubleValidator

import cv2
import numpy as np
from ColorMask import ColorMask

class Camera:
    def __init__(
        self,
        cam_index=0,
        name="Camera",
        use_csi=False,
        sensor_id=0,
    ):
        self.color_mask = ColorMask(camera_name=name)
        self.latest_frame = None
        self.target_found = False
        self.last_target_center = None
        self.use_csi = use_csi
        self.sensor_id = sensor_id
        self.cam_index = cam_index

    def start_cap(self):
        if self.use_csi:
            pipeline = (
                "nvarguscamerasrc sensor-id={} ! "
                "video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, "
                "framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
            ).format(self.sensor_id)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.cam_index)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        # Optionally do BGR→RGB if you need that, or just skip if you want identical handling
        self.latest_frame = frame
        self.color_mask.set_frame(frame)

        return True, self.latest_frame


    def detect_target(self):
        self.last_target_center = None  # reset
        if self.latest_frame is None:
            return None, None, False

        mask, overlay, found = self.color_mask.apply(self.latest_frame)
        self.target_found = found

        if found:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                self.last_target_center = (x + w // 2, y + h // 2)

        return mask, overlay, found

    def get_display_frame(self, draw_bbox=True):
        if self.latest_frame is None:
            return None
        display_frame = self.latest_frame.copy()
        if draw_bbox and self.target_found and self.last_target_center:
            x, y = self.last_target_center
            mask, _, found = self.color_mask.apply(self.latest_frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert back to BGR if you need to display it with OpenCV
        return cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

    def get_center_of_mask(self):
        return self.last_target_center

    def open_tuner(self):
        self.color_mask.open_tuner()

    def release(self):
        if self.use_csi:
            # If you are using the CSI_Camera code that spawns a thread:
            # self.cap.stop()       # stop the background thread
            self.cap.release()    # then release pipeline
        else:
            # If it’s just plain cv2.VideoCapture(...) with a GStreamer pipeline
            self.cap.release()

class CameraHandler(QObject):
    frame_ready = Signal(QImage)
    detection_update = Signal(str, bool, tuple)

    def __init__(self, cam_index=0, name="Camera", use_csi=False, sensor_id=0):
        super().__init__()
        self.camera = Camera(cam_index, name, use_csi, sensor_id)
        self.active = False
        self.detection_active = False
        self.name = name

    def start(self):
        self.camera.start_cap()
        self.active = True

    def stop(self):
        self.active = False
        self.camera.release()

    def update_frame(self):
        if not self.active:
            return

        ret, frame = self.camera.read_frame()
        if not ret:
            return

        if self.detection_active:
            self.camera.detect_target()
            if self.camera.target_found and self.camera.last_target_center:
                center = self.camera.last_target_center
                self.detection_update.emit(self.name, True, center)
            else:
                self.detection_update.emit(self.name, False, None)

        display_frame = self.camera.get_display_frame(self.detection_active)
        if display_frame is not None:
            h, w, ch = display_frame.shape
            bytes_per_line = ch * w
            image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(image)

    def toggle_detection(self, active):
        self.detection_active = active

    def open_tuner(self):
        self.camera.open_tuner()

class Gantry(QObject):
    position_updated = Signal(float, float, float)  # Current position
    target_updated = Signal(float, float, float)    # Target position
    
    def __init__(self, port=None, baudrate=9600, timeout=2):
        super().__init__()
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_connected = False
        
        # All gantry state is now encapsulated here
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_z = 0.0
        self._target_x = 50.0  # Default values
        self._target_y = 50.0
        self._target_z = 50.0
        
        if port is not None:
            self.connect(port, baudrate, timeout)

    @property
    def current_position(self):
        return (self._current_x, self._current_y, self._current_z)
        
    @property
    def target_position(self):
        return (self._target_x, self._target_y, self._target_z)
        
    @target_position.setter
    def target_position(self, pos):
        self._target_x, self._target_y, self._target_z = pos
        self.target_updated.emit(*pos)

    def connect(self, port, baudrate=9600, timeout=2):
        """Thread-safe connection with proper cleanup"""
        try:
            # Skip if already connected to this port
            if self.is_connected and self.port == port:
                return True
                
            # Cleanup existing connection
            self.disconnect()
            
            # Validate port
            if not port or not isinstance(port, str):
                return False
                
            # Attempt connection
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(1)  # Reduced wait time
            self.port = port
            self.is_connected = True
            return True
            
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.is_connected = False
            return False
        
    def move_to(self, x, y, z):
        """Send movement command to gantry and update target position"""
        if not self.is_connected:
            print("Gantry not connected!")
            return False
            
        try:
            cmd = f"GOTO {x:.2f} {y:.2f} {z:.2f}\n"
            self.ser.write(cmd.encode("utf-8"))
            print(f"Sent: {cmd.strip()}")
            self.target_position = (x, y, z)  # Update target and emit signal
            return True
        except Exception as e:
            print(f"Error sending gantry command: {e}")
            return False
        
    def disconnect(self):
        """Properly disconnect from the serial port"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False

    def home(self):
        """Send homing command to gantry"""
        if not self.is_connected:
            return False
        try:
            self.ser.write(b"HOME\n")
            return True
        except Exception as e:
            print(f"Homing failed: {e}")
            return False

    def read_position(self):
        """Request and parse current position from hardware"""
        if not self.is_connected:
            return None
        try:
            self.ser.write(b"GETPOS\n")
            response = self.ser.readline().decode().strip()
            if response.startswith("POS:"):  # Example format
                x, y, z = map(float, response.split()[1:4])
                self._current_x, self._current_y, self._current_z = x, y, z
                self.position_updated.emit(x, y, z)
                return (x, y, z)
        except Exception as e:
            print(f"Position read failed: {e}")
        return None
    
class EndEffector(QObject):
    position_updated = Signal(float, float)  # Current position (theta, x)
    target_updated = Signal(float, float)    # Target position (theta, x)
    
    def __init__(self, port=None, baudrate=9600, timeout=2):
        super().__init__()
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_connected = False
        
        # End effector state
        self._current_theta = 0.0  # Rotation angle (degrees)
        self._current_x = 0.0       # Linear position (mm)
        self._target_theta = 90.0   # Default values (middle positions)
        self._target_x = 50.0
        
        if port is not None:
            self.connect(port, baudrate, timeout)

    @property
    def current_position(self):
        return (self._current_theta, self._current_x)
        
    @property
    def target_position(self):
        return (self._target_theta, self._target_x)
        
    @target_position.setter
    def target_position(self, pos):
        self._target_theta, self._target_x = pos
        self.target_updated.emit(*pos)

    def connect(self, port, baudrate=9600, timeout=2):
        """Thread-safe connection with proper cleanup"""
        try:
            # Skip if already connected to this port
            if self.is_connected and self.port == port:
                return True
                
            # Cleanup existing connection
            self.disconnect()
            
            # Validate port
            if not port or not isinstance(port, str):
                return False
                
            # Attempt connection
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(1)  # Reduced wait time
            self.port = port
            self.is_connected = True
            return True
            
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.is_connected = False
            return False
        
    def move_to(self, theta, x):
        """Send movement command to end effector and update target position"""
        if not self.is_connected:
            print("End effector not connected!")
            return False
            
        try:
            cmd = f"EEGOTO {theta:.2f} {x:.2f}\n"
            self.ser.write(cmd.encode("utf-8"))
            print(f"Sent: {cmd.strip()}")
            self.target_position = (theta, x)  # Update target and emit signal
            return True
        except Exception as e:
            print(f"Error sending end effector command: {e}")
            return False
        
    def disconnect(self):
        """Properly disconnect from the serial port"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False

    def home(self):
        """Send homing command to end effector"""
        if not self.is_connected:
            return False
        try:
            self.ser.write(b"EEHOME\n")
            return True
        except Exception as e:
            print(f"Homing failed: {e}")
            return False

    def read_position(self):
        """Request and parse current position from hardware"""
        if not self.is_connected:
            return None
        try:
            self.ser.write(b"EEGETPOS\n")
            response = self.ser.readline().decode().strip()
            if response.startswith("EEPOS:"):  # Example format
                theta, x = map(float, response.split()[1:3])
                self._current_theta, self._current_x = theta, x
                self.position_updated.emit(theta, x)
                return (theta, x)
        except Exception as e:
            print(f"Position read failed: {e}")
        return None

class ConnectionWorker(QObject):
    finished = Signal(bool)  # Success status
    
    def __init__(self, gantry, port):
        super().__init__()
        self.gantry = gantry
        self.port = port
    
    def run(self):
        """Thread-safe connection attempt"""
        try:
            result = self.gantry.connect(self.port)
            self.finished.emit(result)
        except Exception as e:
            print(f"Connection error: {e}")
            self.finished.emit(False)