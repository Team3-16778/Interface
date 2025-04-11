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
        """Connect to the specified serial port"""
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
                
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Wait for Arduino to initialize
            self.port = port
            self.baudrate = baudrate
            self.timeout = timeout
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to gantry: {e}")
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

class SerialReaderThread(QThread):
    data_received = Signal(str)
    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.running = True

    def run(self):
        while self.running:
            if self.serial_port.in_waiting:
                try:
                    data = self.serial_port.readline().decode('utf-8').strip()
                    if data:
                        self.data_received.emit(data)
                except Exception as e:
                    print("Error reading serial:", e)
            self.msleep(1)

    def stop(self):
        self.running = False
        self.wait()

class SerialSenderThread(QThread):
    def __init__(self, arduino, command):
        super().__init__()
        self.arduino = arduino
        self.command = command

    def run(self):
        if self.arduino.is_open:
            self.arduino.write((self.command + "\n").encode("utf-8"))

