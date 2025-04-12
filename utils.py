import sys
import re
import sys
import re
import time
import serial
from serial.tools import list_ports
import cv2
import numpy as np

from abc import ABC, abstractmethod

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QLCDNumber, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont, QDoubleValidator

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

class AbstractSerialDevice(ABC):
    def __init__(self, port="/dev/ttyACM0", baud=9600, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self._open_serial()

    def _open_serial(self):
        """(Re)open the serial connection using current settings."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=self.timeout
        )
        time.sleep(2)  # Allow device time to reset/listen

    def set_port(self, port, baud=None, timeout=None):
        """Change the port (and optionally baud/timeout) at runtime."""
        self.port = port
        if baud is not None:
            self.baud = baud
        if timeout is not None:
            self.timeout = timeout
        self._open_serial()

    @abstractmethod
    def goto_position(self, *args):
        """
        Move this device to the specified position (args vary by subclass).
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Send a 'STOP' or emergency stop command to the device.
        """
        pass

class Gantry(AbstractSerialDevice):
    def __init__(self, port="/dev/ttyACM0", baud=9600, timeout=1.0):
        super().__init__(port=port, baud=baud, timeout=timeout)
        print(f"Initialized Gantry on {port} at {baud} baud.")

    def goto_position(self, x, y, z):
        """Send GOTO command to move gantry to (x, y, z)."""
        cmd = f"GOTO {x:.2f} {y:.2f} {z:.2f}\n"
        self.ser.write(cmd.encode())
        print(f"SENT: {cmd.strip()}")

    def stop(self):
        """Send STOP command (emergency stop)."""
        self.ser.write(b"STOP\n")

    def home(self):
        """Send HOME command to home the gantry."""
        self.ser.write(b"HOME\n")

    def get_position(self):
        """
        Send GETPOS command and parse the response, e.g. "POS: 10.0 5.0 7.0".
        Returns a tuple (x, y, z) or None if parsing fails.
        """
        self.ser.write(b"GETPOS\n")
        response = self.ser.readline().decode().strip()
        if response.startswith("POS:"):
            coords = response[4:].split()
            if len(coords) == 3:
                x, y, z = map(float, coords)
                return (x, y, z)
        return None


class EndEffector(AbstractSerialDevice):
    def __init__(self, port="/dev/ttyACM1", baud=9600, timeout=1.0):
        super().__init__(port=port, baud=baud, timeout=timeout)
        print(f"Initialized EndEffector on {port} at {baud} baud.")

    def goto_position(self, theta, delta):
        """Example command to rotate by two angles."""
        cmd = f"ROTATE {theta:.2f} {delta:.2f}\n"
        self.ser.write(cmd.encode())
        print(f"SENT: {cmd.strip()}")

    def stop(self):
        self.ser.write(b"STOP\n")

    def get_angles(self):
        """
        Example command to read angles back from the device.
        Adjust as needed to match your firmware.
        """
        self.ser.write(b"GETANGLES\n")
        response = self.ser.readline().decode().strip()
        if response.startswith("ANGLES:"):
            vals = response[7:].split()
            if len(vals) == 2:
                t, d = map(float, vals)
                return (t, d)
        return None
