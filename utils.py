import sys
import re
import sys
import re
import time
import serial
from serial.tools import list_ports
import cv2
import numpy as np
import platform


from abc import ABC, abstractmethod

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QLCDNumber, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, Slot, QMutex, QMutexLocker
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
        self.processing_active = False  # New flag for selective processing
        self._is_capturing = False
        self.cap = None

    def start_cap(self):
        if self._is_capturing:
            return
            
        if self.use_csi:
            # Optimized pipeline with lower resolution
            pipeline = (
                "nvarguscamerasrc sensor-id={} ! "
                "video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, "
                "framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
            ).format(self.sensor_id)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.cam_index)
            
        self._is_capturing = True

    def read_frame(self):
        if not self._is_capturing or self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        
        self.latest_frame = frame
        if self.processing_active:
            self.color_mask.set_frame(frame)
            
        return True, self.latest_frame

    def release(self):
        if not self._is_capturing:
            return
            
        if self.cap is not None:
            if self.use_csi:
                # Special handling for CSI cameras
                self.cap.release()
                # Add small delay to ensure resources are freed
                time.sleep(0.1)
            else:
                self.cap.release()
                
            self.cap = None
            
        self._is_capturing = False
        self.latest_frame = None

    def detect_target(self):
        if not self.processing_active or self.latest_frame is None:
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
            # Get the bounding box dimensions from color mask
            mask, _, _ = self.color_mask.apply(self.latest_frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy = x + w // 2, y + h // 2
                
                # Draw the bounding box and center point (same as ColorMask)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display_frame, (cx, cy), 5, (255, 255, 0), -1)
                
                # Update the last target center
                self.last_target_center = (cx, cy)

        return cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

    def set_processing_active(self, active):
        """Enable/disable all color processing"""
        self.processing_active = active

    def get_center_of_mask(self):
        return self.last_target_center

    def open_tuner(self):
        self.color_mask.open_tuner()

class CameraHandler(QObject):
    frame_ready = Signal(QImage)
    detection_update = Signal(str, bool, tuple)
    
    def __init__(self, cam_index=0, name="Camera", use_csi=False, sensor_id=0):
        super().__init__()
        self.camera = Camera(cam_index, name, use_csi, sensor_id)
        self.active = False
        self.detection_active = True
        self.name = name
        self.frame_counter = 0
        self.detection_interval = 3
        self.display_interval = 1
        self.last_center = None
        self.gui_active = False
        self._mutex = QMutex()  # Add mutex for thread safety

    def start(self):
        with QMutexLocker(self._mutex):
            self.camera.start_cap()
            self.active = True
        
    def stop(self):
        with QMutexLocker(self._mutex):
            self.active = False
            self.camera.release()

    def update_frame(self):
        if not self.active:
            return
            
        with QMutexLocker(self._mutex):
            if not self.camera._is_capturing:
                return
                
            self.frame_counter += 1
            ret, frame = self.camera.read_frame()
            if not ret:
                return

            # Process detection only when active and interval met
            if self.detection_active and (self.frame_counter % self.detection_interval == 0):
                self.camera.detect_target()
                if self.camera.target_found and self.camera.last_target_center:
                    self.last_center = self.camera.last_target_center
                    self.detection_update.emit(self.name, True, self.last_center)
                else:
                    self.detection_update.emit(self.name, False, None)

            # Update display more frequently than detection
            if self.frame_counter % self.display_interval == 0 and self.gui_active:
                display_frame = self.camera.get_display_frame(self.detection_active)
                if display_frame is not None:
                    h, w, ch = display_frame.shape
                    bytes_per_line = ch * w
                    image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.frame_ready.emit(image)

    def toggle_detection(self, active):
        self.detection_active = active
        self.camera.set_processing_active(active)  # Propagate to camera
        
        # Reset counter when toggling to maintain interval timing
        if active:  
            self.frame_counter = 0

    def set_detection_interval(self, interval):
        """Dynamically adjust how often detection runs"""
        self.detection_interval = max(1, interval)  # Minimum 1 frame
        
    def open_tuner(self):
        # Enable full processing while tuner is open
        self.camera.set_processing_active(True)
        self.camera.open_tuner()
        

class AbstractSerialDevice(ABC):
    def __init__(self, port="/dev/ttyACM0", baud=9600, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

        try:
            if self.port:
                self._open_serial()
            else:
                print("No port specified. Serial connection not opened.")
        except Exception as e:
            print(f"Failed to open serial connection: {e}")

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

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0

        time.sleep(2)

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

    def get_target(self):
        """
        Return the target position as a tuple (x, y, z).
        """
        return (self.target_x, self.target_y, self.target_z)
    
    def set_target(self, x, y, z):
        """
        Set the target position for the gantry.
        """
        self.target_x = x
        self.target_y = y
        self.target_z = z

    def send_to_target(self):
        """
        Send the current target position to the gantry.
        """
        self.goto_position(self.target_x, self.target_y, self.target_z)

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
class HardwareManager:
    def __init__(self):
        self.camera1: CameraHandler | None = None
        self.camera2: CameraHandler | None = None
        self.gantry: Gantry | None = None
        self.end_effector: EndEffector | None = None
        
        # Camera to world coordinates conversion
        self.camera_center_x = 320  # Assuming 640x480 resolution
        self.pixels_to_mm = 0.5

        self.connect_cameras()
        self.connect_gantry()
        self.connect_effector()
    
    def connect_gantry(self):
        """Initialize gantry with basic error handling"""
        try:
            self.gantry = Gantry()
        except Exception as e:
            print(f"Gantry init failed: {e}")
        
    def connect_effector(self):
        """Initialize end effector with basic error handling"""
        try:
            self.effector = EndEffector()
        except Exception as e:
            print(f"End Effector init failed: {e}")

    def connect_cameras(self):
        """Initialize cameras with basic error handling"""
        try:
            is_mac = platform.system() == "Darwin"
            self.camera1 = CameraHandler(0, "Camera 1", use_csi=not is_mac, sensor_id=0)
            self.camera2 = CameraHandler(1, "Camera 2", use_csi=not is_mac, sensor_id=1)
            return True
        except Exception as e:
            print(f"Camera init failed: {e}")
            return False
    
    def close_all(self):
        """Cleanup all connections"""
        if self.gantry:
            self.gantry.stop()
        if self.camera1:
            self.camera1.stop()
            time.sleep(0.5)
        if self.camera2:
            self.camera2.stop()
            time.sleep(0.5)

        cv2.destroyAllWindows()
    
    def pid_x_axis(self):
        """Update gantry x position based on camera's y-axis target position"""
        if not self.camera1 or not self.gantry:
            return
        
        # Get current target from camera
        target = self.camera1.camera.get_center_of_mask()
        if not target or not self.camera1.camera.target_found:
            print("No target detected")
            return  # No target detected
        
        try:
            _, target_y = target
            current_x, current_y, current_z = self.gantry.get_position()
            
            # Update gantry X position based on camera Y position
            new_x = (240 - target_y) * self.pixels_to_mm
            new_x = (540 - target_y)

            
            print(f"Moving gantry to X: {new_x:.2f} based on camera Y: {target_y}")
            self.gantry.set_target(new_x, current_y, current_z)
            self.gantry.send_to_target()
            
        except Exception as e:
            print(f"Error in PID control: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = HardwareManager()

    # Calibration parameters
    manager.pixels_to_mm = 0.1  # Adjust based on your setup
    
    # Start cameras with processing enabled
    manager.camera1.start()
    manager.camera1.camera.processing_active = True
    manager.camera1.detection_active = True

    manager.camera1.open_tuner()

    def update_and_control():
        manager.camera1.update_frame()
        manager.pid_x_axis()
        target = manager.camera1.camera.get_center_of_mask()
        if target:
            print(f"Camera Y position: {target[1]}")
        else:
            print("No target detected")

    timer = QTimer()
    timer.timeout.connect(update_and_control)
    timer.start(10)

    sys.exit(app.exec())