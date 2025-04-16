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
        intrinsics=None,
        extrinsics = None
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

        self.camera_intrinsics = intrinsics
        self.camera_extrinsics = extrinsics

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
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
    
    def get_world_3d(self, u, v, Zc = 0.6731): # Zc is the distance from the camera to the target(default is Acotag Distance~0.6731)
        """
        Calculate the 3D coordinates of the target in the world frame
        Input:
            u: x coordinate of the target in the image
            v: y coordinate of the target in the image
            Zc: depth of the target in the camera frame (distance from the camera to the target)
        Output:
            world_point: 3D coordinates of the target in the world frame
        """
        # Get the parameters we need: focal length, principal point, and rotation vector and translation vector
        camera_matrix = self.camera_intrinsics["camera_matrix"]
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        Rotation_matrix = self.camera_extrinsics[0:3, 0:3]
        Translation_vector = self.camera_extrinsics[0:3, 3].reshape(3, 1)

        # 4. Compute the 3D coordinates of the target in the camera frame
        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy
        camera_point = np.array([[Xc], [Yc], [Zc]])

        # 5. Compute the 3D coordinates of the target in the world frame
        # Method 1: external parameters is T_world_camera
        world_point = Rotation_matrix @ camera_point + Translation_vector

        # Method 2: external parameters is T_camera_world
        # world_point = Rotation_matrix.T @ (camera_point - Translation_vector)

        return world_point.flatten()

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
        self.detection_interval = 5
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

    def home(self):
        """Send HOME command."""
        try:
            if self.ser is not None:
                print("Homing device...")
                self.ser.write(b"HOME\n")
            else:
                print("Warning: Serial connection is None, cannot send HOME")
        except Exception as e:
            print(f"Error while sending HOME command: {e}")
    
    def inject(self):
        """Send INJECT command."""
        try:
            if self.ser is not None:
                print("Injecting...")
                self.ser.write(b"INJECT\n")
            else:
                print("Warning: Serial connection is None, cannot send INJECT")
        except Exception as e:
            print(f"Error while sending INJECT command: {e}")

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
        try:
            cmd = f"GOTO {int(x)} {int(y)} {int(z)}\n"
            self.ser.write(cmd.encode())
            print(f"SENT: {cmd.strip()}")
        except Exception as e:
            print(f"Error while sending GOTO command: {e}")

    def stop(self):
        if self.ser is not None:
            self.ser.write(b"STOP\n")
        else:
            print("Warning: Serial connection is None, cannot send STOP")

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

    def injectA(self):
        """Send INJECT A command."""
        try:
            if self.ser is not None:
                print("Injecting...")
                self.ser.write(b"INJECTA\n")
            else:
                print("Warning: Serial connection is None, cannot send INJECTA")
        except Exception as e:
            print(f"Error while sending INJECTA command: {e}")

    def injectC(self):
        """Send INJECT command."""
        try:
            if self.ser is not None:
                print("Injecting...")
                self.ser.write(b"INJECTC\n")
            else:
                print("Warning: Serial connection is None, cannot send INJECTC")
        except Exception as e:
            print(f"Error while sending INJECTC command: {e}")

class EndEffector(AbstractSerialDevice):
    def __init__(self, port="/dev/ttyACM1", baud=9600, timeout=1.0):
        super().__init__(port=port, baud=baud, timeout=timeout)
        print(f"Initialized EndEffector on {port} at {baud} baud.")

    def goto_position(self, theta, delta):
        """Example command to rotate by two angles."""
        try:
            cmd = f"ROTATE {theta:.2f} {delta:.2f}\n"
            self.ser.write(cmd.encode())
            print(f"SENT: {cmd.strip()}")
        except Exception as e:
            print(f"Error while sending ROTATE command: {e}")

    def stop(self):
        if self.ser is not None:
            self.ser.write(b"STOP\n")
        else:
            print("Warning: Serial connection is None, cannot send STOP")

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

        self.connect_cameras()
        self.connect_gantry()
        self.connect_effector()

        # Blind control params -- buggy and should be removed upon teammate completion on arduino comm reliability
        self.current_x = 0.0 
        self.current_y = 0.0
        self.current_z = 0.0

    def set_blind_vals(self, x, y, z):
        """Set blind control values"""
        self.current_x = x
        self.current_y = y
        self.current_z = z
        self.gantry.set_target(x, y, z)

    
    def connect_gantry(self):
        """Initialize gantry with basic error handling"""
        try:
            self.gantry = Gantry()
        except Exception as e:
            print(f"Gantry init failed: {e}")
        
    def connect_effector(self):
        """Initialize end effector with basic error handling"""
        try:
            self.end_effector = EndEffector()
        except Exception as e:
            print(f"End Effector init failed: {e}")

    def connect_cameras(self):
        """Initialize cameras with basic error handling"""
        try:
            is_mac = platform.system() == "Darwin"
            is_windows = platform.system() == "Windows"
            is_jetson = not is_mac and not is_windows  # Assume Jetson if not Mac or Windows

            self.camera1 = CameraHandler(0, "Camera 1", use_csi=is_jetson, sensor_id=0)
            self.camera2 = CameraHandler(1, "Camera 2", use_csi=is_jetson, sensor_id=1)
            return True
        except Exception as e:
            print(f"Camera init failed: {e}")
            return False
    
    def close_all(self):
        """Cleanup all connections"""
        print("Starting shutdown sequence...")
        
        # Stop gantry first if moving
        if self.gantry:
            print("Stopping gantry...")
            self.gantry.stop()
        
        # Special handling for CSI cameras
        if self.camera1 and self.camera1.camera.use_csi:
            print("Shutting down Camera 1 (CSI)...")
            self.camera1.stop()
            time.sleep(1.0)  # Extra time for CSI
            
        if self.camera2 and self.camera2.camera.use_csi:
            print("Shutting down Camera 2 (CSI)...")
            self.camera2.stop()
            time.sleep(1.0)  # Extra time for CSI
        
        # Regular camera cleanup
        if self.camera1 and not self.camera1.camera.use_csi:
            print("Shutting down Camera 1...")
            self.camera1.stop()
            
        if self.camera2 and not self.camera2.camera.use_csi:
            print("Shutting down Camera 2...")
            self.camera2.stop()
        
        # Additional cleanup
        print("Final cleanup...")
        cv2.destroyAllWindows()
        
        # Extra garbage collection
        import gc
        gc.collect()

        # Restart camera module
        import os
        os.system('systemctl restart nvargus-daemon')
        time.sleep(2.0)
        
        print("Shutdown complete")

    def x_axis_controller(self):
        """Simplified control: move left if target is right of center, right if left of center."""
        if not self.camera1 or not self.gantry:
            return

        target = self.camera1.camera.get_center_of_mask()
        if not target or not self.camera1.camera.target_found:
            print("No target detected")
            return

        try:
            _, target_y = target
            center_y = 240  # Midpoint of 480px frame

            pos = self.gantry.get_position()
            if not pos:
                print("Could not read gantry position")
                return
            current_x, current_y, current_z = pos

            step_mm = 3.0  # Fixed step size
            if abs(target_y - center_y) < 10:
                print("Target centered.")
                return
            elif target_y > center_y:
                new_x = current_x - step_mm  # Move left
            else:
                new_x = current_x + step_mm  # Move right

            print(f"Step move to X: {new_x:.2f} (Target Y: {target_y})")
            self.gantry.set_target(new_x, current_y, current_z)
            self.gantry.send_to_target()

        except Exception as e:
            print(f"Error in simple controller: {e}")

    ## Teammate isn't here to implement an effective return value for gantry.get_position() so we send it
    def blind_x_control(self):
        if not self.camera1 or not self.gantry:
            return

        target = self.camera1.camera.get_center_of_mask()
        if not target or not self.camera1.camera.target_found:
            print("No target detected")
            return

        try:
            _, target_y = target
            center_y = 240
            step_mm = 2.0

            if abs(target_y - center_y) < 10:
                print("Target centered.")
                return
            elif target_y > center_y:
                self.current_x -= step_mm
            else:
                self.current_x += step_mm

            print(f"Blind step to X: {self.current_x:.2f} (Target Y: {target_y})")
            self.gantry.set_target(self.current_x, self.current_y, self.current_z)
            self.gantry.send_to_target()

        except Exception as e:
            print(f"Error in blind controller: {e}")
    
    def send_yz_position(self, y, z):
        """Send Y and Z commands to the gantry while keeping X fixed."""
        self.current_y = y
        self.current_z = z
        self.gantry.set_target(self.current_x, y, z)
        self.gantry.send_to_target()

    def send_theta_to_effector(self, theta, delta=0.0):
        """Send rotation command to end effector."""
        if self.end_effector:
            self.end_effector.goto_position(theta, delta)

    def inject_all(self):
        """Trigger injection on both gantry and end effector."""
        if self.gantry:
            self.gantry.inject()
        if self.end_effector:
            self.end_effector.inject()

    def home_all(self):
        """Home all devices."""
        if self.gantry:
            self.gantry.home()
        if self.end_effector:
            self.end_effector.home()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = HardwareManager()
    # Print ports on both devices
    print("Gantry Port:", manager.gantry.port)
    print("End Effector Port:", manager.end_effector.port)
    manager.home_all()
    time.sleep(35)  # Allow time for homing
    print("Homing complete. Starting camera processing...")

    manager.gantry.goto_position(230, 260, 140)  # Move gantry to preset position for X calibration
    manager.set_blind_vals(230, 260, 140)  # Set blind control values


    # Load calibration files
    manager.camera1.camera.camera_intrinsics = np.load('camera2_calibration_data.npz')
    manager.camera1.camera.camera_extrinsics = np.load('cam2_external_parameters_0415.npz')["T_world_camera"]
    manager.camera2.camera.camera_intrinsics = np.load('camera2_calibration_data.npz')
    manager.camera2.camera.camera_extrinsics = np.load('cam2_external_parameters_0415.npz')["T_world_camera"]

    # Start cameras with processing enabled
    manager.camera1.start()
    manager.camera1.camera.processing_active = True
    manager.camera1.detection_active = True
    manager.camera1.open_tuner()

    manager.camera2.start()
    manager.camera2.camera.processing_active = True
    manager.camera2.detection_active = True

    alignment_active = False  # Global flag to enable X control during alignment


    time.sleep(5)


    
    # Global camera update + control loop
    TARGET_UPDATE_INTERVAL = 0.5  # seconds
    last_target_time = 0.0

    last_x_control_time = 0.0
    X_CONTROL_INTERVAL = 0.5  # seconds

    def update_and_control():
        global last_target_time, last_x_control_time
        now = time.time()

        manager.camera1.update_frame()

        if alignment_active and (now - last_x_control_time) >= X_CONTROL_INTERVAL:
            manager.blind_x_control()
            last_x_control_time = now

        if (now - last_target_time) >= TARGET_UPDATE_INTERVAL:
            target = manager.camera1.camera.get_center_of_mask()
            if target:
                u, v = target
                world_3d = manager.camera1.camera.get_world_3d(u, v)
                print("3D position:", world_3d)
            else:
                print("No target detected")
            last_target_time = now

    timer = QTimer()
    timer.timeout.connect(update_and_control)
    timer.start(500)

    # Main automation sequence
    def run_full_automation_sequence():
        print("\n[Sequence Start] Aligning X, then moving to Y/Z, rotating theta, then injecting.")

        def step1_align_x():
            global alignment_active
            alignment_active = True
            print("Step 1: Starting X-axis alignment.")
            start_time = time.time()
            check_alignment_progress(start_time)

        def check_alignment_progress(start_time):
            global alignment_active
            target = manager.camera1.camera.get_center_of_mask()
            timeout = 25  # seconds

            if target and manager.camera1.camera.target_found:
                _, target_y = target
                center_y = 240
                if abs(target_y - center_y) < 10:
                    print("Target aligned — moving to next step.")
                    alignment_active = False
                    step2_yz_position()
                    return

            if time.time() - start_time > timeout:
                print("Alignment timeout — moving to next step anyway.")
                alignment_active = False
                step2_yz_position()
                return

            # Retry after short delay
            QTimer.singleShot(300, lambda: check_alignment_progress(start_time))


        def step2_yz_position():
            print("Step 2: Sending Y/Z position phase 1.")
            global alignment_active
            alignment_active = False

            def calculate_yz():
                # # Get X position from gantry
                # dis_x = manager.gantry.get_position()[0]
                # offset_x = 50 #mm, ~2 inch(from camera to gantry x zero)
                # dis_x = dis_x + offset_x
                # target = manager.camera2.camera.get_center_of_mask()
                target = None
                if target:
                    u, v = target
                    # calculate 3D position of the target in world frame
                    target_y, target_z, _ = manager.camera2.camera.get_world_3d(u, v, dis_x)
                else: # if no target detected (go to backup position: y is at the center of the camera2)
                    print("Target not detected, using backup position.")
                    target_y = 0.4826 - 172.8/1000 # preset 0.4826m(19 inches) from gantry y zero
                    target_z = -0.4492625 + 0.1778 # preset 0.1778m(7 inches): target from ground(-0.4492625m in world calculation)

                L_ext = 275.72 #mm, from end effector rotation center to external pin
                ET = L_ext + 30.0 + 100.0 # preset 100mm from pin to external pin to target center, 30mm for internal pin
                theta = 120*np.pi/180 # preset theta for end effector

                gantry_offset_y = 172.8 #mm, 11-4 inch (-5mm for safety)
                target_y = target_y*1000 + gantry_offset_y
                gantry_des_y = target_y - ET*np.sin(theta)

                gantry_offset_z = 7.7 #mm, 2-1.5 inch (-5mm for safety)
                target_z = - target_z*1000 + gantry_offset_z
                gantry_des_z = target_z + ET*np.cos(theta)

                print("The desired Y and Z positions for gantry are: {}, {}".format(gantry_des_y, gantry_des_z))
                return gantry_des_y, gantry_des_z
            
            gantry_des_y, gantry_des_z = calculate_yz()

            print("Step 2: Sending Y/Z position.")
            manager.send_yz_position(y=gantry_des_y, z=gantry_des_z)
            QTimer.singleShot(10000, step3_theta)

        def step3_theta():
            print("Step 3: Sending theta to end effector.")
            manager.send_theta_to_effector(theta=120.0, delta=0.0)
            QTimer.singleShot(10000, step4a_inject_gantry)

        def step4a_inject_gantry():
            print("Step 4a: Injecting gantry.")
            manager.gantry.injectA()
            QTimer.singleShot(10000, step4b_inject_both)

        def step4b_inject_both():
            print("Step 4b: Injecting both.")
            manager.inject_all()
            QTimer.singleShot(10000, step4c_retract_sample)

        def step4c_retract_sample():
            print("Step 4c: Retracting sample.")
            manager.gantry.injectC()
            QTimer.singleShot(10000, step5_home)

        def step5_home():
            print("Step 5: Homing gantry and end effector.")
            manager.home_all()

        step1_align_x()

    # Start sequence after app init
    QTimer.singleShot(0, run_full_automation_sequence)

    # Graceful shutdown on Ctrl+C
    def run_app():
        try:
            sys.exit(app.exec())
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Closing all connections...")
            manager.close_all()
            QTimer.singleShot(0, app.quit)

    import signal
    def handle_interrupt(signum, frame):
        print("\nKeyboardInterrupt (signal) detected. Closing all connections...")
        manager.close_all()
        QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, handle_interrupt)

    run_app()