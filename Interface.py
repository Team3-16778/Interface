import os
import sys
import re
import serial
from serial.tools import list_ports
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QLCDNumber, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, Signal
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont, QDoubleValidator
from camera_utils import CSI_Camera, gstreamer_pipeline, colormask, calculate_world_3D
from ColorMask import ColorMask
import cv2
import numpy as np

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
        # # gstreamer parameters
        # self.capture_width=capture_width,
        # self.capture_height=capture_height,
        # self.display_width=display_width,
        # self.display_height=display_height,
        # self.framerate=framerate,
        # self.flip_method=flip_method        
        # Internal Parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        # External Parameters
        self.Rotation_matrix = None
        self.Translation_vector = None

    def start_cap(self):
        if self.use_csi:
            pipeline = (
                "nvarguscamerasrc sensor-id={} ! "
                "video/x-raw(memory:NVMM), width={}, height={}, format=(string)NV12, "
                "framerate=(fraction){}/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
            ).format(self.sensor_id, self.capture_width, self.capture_height, self.framerate)
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

    def get_internal_parameters(self, file_name):
        camera_internal_file = dir_path + "/" + file_name
        internal_data = np.load(camera_internal_file)
        self.camera_matrix = internal_data["camera_matrix"]
        self.dist_coeffs = internal_data["dist_coeffs"]
        print("Camera Matrix: \n", self.camera_matrix)
        print("Distortion Coefficients: \n", self.dist_coeffs)

    def get_external_parameters(self, file_name):
        camera_external_file = dir_path + "/" + file_name
        T_world_camera = np.load(camera_external_file)["T_world_camera"]
        self.Rotation_matrix = T_world_camera[0:3, 0:3]
        self.Translation_vector = T_world_camera[0:3, 3].reshape(3, 1)    

    def calculate_center_distance(self, u, v):
        """
        Calculate the distance from the target point to the camera center
        """
        if self.camera_matrix is None:
            print("No internal parameters!!!")
            return None, None
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        # print(f"Center(Pixel): {cx} {cy}")        
        return u-cx, v-cy
    
    def calculate_world_3D(self, u, v, Zc = 0.5):# Zc is the distance from the camera to the target(default is a random value)
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
        if (self.camera_matrix is None or self.Rotation_matrix is None or self.Translation_vector is None):
            return None
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        Rotation_matrix = self.Rotation_matrix
        Translation_vector = self.Translation_vector

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


class GantryController:
    def __init__(self, port, baudrate=9600, timeout=2):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for Arduino to initialize

    def move_to(self, x, y, z):
        cmd = f"GOTO {x:.2f} {y:.2f} {z:.2f}\n"
        self.ser.write(cmd.encode("utf-8"))
        print(f"Sent: {cmd.strip()}")

    def close(self):
        self.ser.close()

class RobotControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()


        ## ------------------ Inner Parameters ------------------
        self.gantry_pos_des_x = 50
        self.gantry_pos_des_y = 50
        self.gantry_pos_des_z = 50
        self.endeff_pos_des_servo = 90
        self.endeff_pos_des_linear = 50

        self.gantry_X = None
        self.gantry_Y = None
        self.gantry_Z = None
        self.endeff_servo = None
        self.endeff_linear = None

        self.target_X = None
        self.target_Y = None
        self.target_Z = None
        self.ribcage_Y = None
        self.ribcage_Z = None

        ## ------------------ Window Setup ------------------
        self.setWindowTitle("Discount daVinci Control Interface")
        # Set window size to maximum available screen geometry
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.resize(screen_geometry.width(), screen_geometry.height())
        
        # Main widget and main layout (horizontal: left and right)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left side layout: split vertically into top (control panels) and bottom (button groups)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        
        # Right side layout: camera displays arranged vertically, with camera toggle button on top
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        # Create fonts for labels and buttons
        font_button_1 = QFont()
        font_button_1.setPointSize(16)
        font_button_1.setBold(True)
        
        font_button_2 = QFont()
        font_button_2.setPointSize(12)
        font_button_2.setBold(True)

        font_label = QFont()
        font_label.setPointSize(14)
        font_label.setBold(True)

        # ------------------ Right: Color Mask Initalization ------------------

        # ------------------ Right: Camera Toggle and Displays ------------------

        # Camera 1 Group
        cam1_group = QGroupBox("Camera 1")
        cam1_group.setFont(font_label)
        cam1_layout = QVBoxLayout()
        cam1_group.setLayout(cam1_layout)
        self.cam_label1 = QLabel()
        self.cam_label1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cam_label1.setStyleSheet("background-color: #000;")
        cam1_layout.addWidget(self.cam_label1)
        right_layout.addWidget(cam1_group)

        # Camera 2 Group
        cam2_group = QGroupBox("Camera 2")
        cam2_group.setFont(font_label)
        cam2_layout = QVBoxLayout()
        cam2_group.setLayout(cam2_layout)
        self.cam_label2 = QLabel()
        self.cam_label2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cam_label2.setStyleSheet("background-color: #000;")
        cam2_layout.addWidget(self.cam_label2)
        right_layout.addWidget(cam2_group)

        # Horizontal layout for camera control buttons
        camera_controls_layout = QHBoxLayout()

        self.cam_toggle_btn = QPushButton("Open Cameras")
        self.cam_toggle_btn.setCheckable(True)
        self.cam_toggle_btn.setFont(font_button_1)
        self.cam_toggle_btn.setMaximumHeight(50)
        self.cam_toggle_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        self.cam_toggle_btn.toggled.connect(self.toggle_cameras)
        camera_controls_layout.addWidget(self.cam_toggle_btn, stretch=2)

        self.cam1_tune_btn = QPushButton("Color Mask 1")
        self.cam1_tune_btn.setMaximumHeight(50)
        self.cam1_tune_btn.setFont(font_button_1)
        self.cam1_tune_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        self.cam1_tune_btn.clicked.connect(lambda: self.cam1.open_tuner())
        camera_controls_layout.addWidget(self.cam1_tune_btn, stretch=1)

        self.cam2_tune_btn = QPushButton("Color Mask 2")
        self.cam2_tune_btn.setMaximumHeight(50)
        self.cam2_tune_btn.setFont(font_button_1)
        self.cam2_tune_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        self.cam2_tune_btn.clicked.connect(lambda: self.cam2.open_tuner())
        camera_controls_layout.addWidget(self.cam2_tune_btn, stretch=1)

        right_layout.addLayout(camera_controls_layout)
  
        # ------------------ Left Top: Control Panels ------------------
        control_layout = QHBoxLayout()
        left_layout.addLayout(control_layout)
        
        # Gantry Control Panel
        self.gantry_group = QGroupBox("Gantry Control")
        self.gantry_group.setFont(font_label)
        self.gantry_group.setStyleSheet("QGroupBox * { font-size: 12px; font-weight: normal; }")
        gantry_layout = QGridLayout()
        self.gantry_group.setLayout(gantry_layout)
        # USB port selection and refresh button
        gantry_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.gantry_port_combo = QComboBox()
        self.gantry_port_combo.addItems(["Waiting"])
        gantry_layout.addWidget(self.gantry_port_combo, 0, 1)
        self.gantry_refresh_btn = QPushButton("Refresh\nPorts")
        self.gantry_refresh_btn.clicked.connect(self.update_gantry_ports)
        gantry_layout.addWidget(self.gantry_refresh_btn, 0, 2)
        # Instead of Enable GUI Control, rename to 'Send Positional Command'
        self.gantry_gui_toggle = QPushButton("Send Positional Command")
        self.gantry_gui_toggle.setMaximumHeight(50)
        self.gantry_gui_toggle.setCheckable(True)
        self.gantry_gui_toggle.setChecked(False)
        self.gantry_gui_toggle.toggled.connect(self.toggle_gantry_gui)
        gantry_layout.addWidget(self.gantry_gui_toggle, 1, 0, 1, 3)
        # Add sliders and LCD displays for three steppers
        self.gantry_sliders = []
        self.gantry_lcds = []
        labels = ["Gantry X", "Gantry Y", "Gantry Z"]
        for i in range(3):
            row = i + 2

            label = QLabel(labels[i])
            label.setFixedWidth(60)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setMinimumWidth(150)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            slider.valueChanged.connect(lambda value, idx=i: self.update_gantry(idx, value))
            self.gantry_sliders.append(slider)

            lcd = QLCDNumber()
            lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
            lcd.display(50)
            lcd.setFixedWidth(60)

            self.gantry_lcds.append(lcd)
            slider.valueChanged.connect(lcd.display)

            gantry_layout.addWidget(label, row, 0)
            gantry_layout.addWidget(slider, row, 1)
            gantry_layout.addWidget(lcd, row, 2)
        gantry_layout.setColumnStretch(0, 0)
        gantry_layout.setColumnStretch(1, 1)
        gantry_layout.setColumnStretch(2, 0)

        control_layout.addWidget(self.gantry_group, alignment=Qt.AlignmentFlag.AlignTop)
        
        # End-Effector Control Panel
        self.endeff_group = QGroupBox("End-Effector Control")
        self.endeff_group.setFont(font_label)
        self.endeff_group.setStyleSheet("QGroupBox * { font-size: 12px; font-weight: normal; }")
        endeff_layout = QGridLayout()
        self.endeff_group.setLayout(endeff_layout)
        # USB port selection and refresh button
        endeff_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.endeff_port_combo = QComboBox()
        self.endeff_port_combo.addItems(["Waiting"])
        endeff_layout.addWidget(self.endeff_port_combo, 0, 1)
        self.endeff_refresh_btn = QPushButton("Refresh\nPorts")
        self.endeff_refresh_btn.clicked.connect(self.update_endeff_ports)
        endeff_layout.addWidget(self.endeff_refresh_btn, 0, 2)
        # Instead of 'Enable GUI Control', rename to 'Send Positional Command'
        self.endeff_gui_toggle = QPushButton("Send Positional Command")
        self.endeff_gui_toggle.setMaximumHeight(50)
        self.endeff_gui_toggle.setCheckable(True)
        self.endeff_gui_toggle.setChecked(False)
        self.endeff_gui_toggle.toggled.connect(self.toggle_endeff_gui)
        endeff_layout.addWidget(self.endeff_gui_toggle, 1, 0, 1, 3)
        # Servo control slider
        servo_label = QLabel("Servo\nControl")
        servo_label.setFixedWidth(60)
        endeff_layout.addWidget(servo_label, 2, 0)
        self.servo_slider = QSlider(Qt.Orientation.Horizontal)
        self.servo_slider.setRange(0, 180)
        self.servo_slider.setValue(90)
        self.servo_slider.setMinimumWidth(150)
        self.servo_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.servo_slider.valueChanged.connect(lambda value: self.update_endeff("servo", value))
        self.servo_lcd = QLCDNumber()
        self.servo_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.servo_lcd.display(90)
        self.servo_lcd.setFixedWidth(60)
        self.servo_slider.valueChanged.connect(self.servo_lcd.display)
        endeff_layout.addWidget(self.servo_slider, 2, 1)
        endeff_layout.addWidget(self.servo_lcd, 2, 2)
        # Linear actuator slider
        syringe_label = QLabel("Syringe\nPosition")
        syringe_label.setFixedWidth(60)
        endeff_layout.addWidget(syringe_label, 3, 0)
        endeff_layout.addWidget(QLabel("Syringe\nPosition"), 3, 0)
        self.linear_slider = QSlider(Qt.Orientation.Horizontal)
        self.linear_slider.setRange(0, 100)
        self.linear_slider.setValue(50)
        self.linear_slider.setMinimumWidth(150)
        self.linear_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.linear_slider.valueChanged.connect(lambda value: self.update_endeff("linear", value))
        self.linear_lcd = QLCDNumber()
        self.linear_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.linear_lcd.display(50)
        self.linear_lcd.setFixedWidth(60)
        self.linear_slider.valueChanged.connect(self.linear_lcd.display)
        endeff_layout.addWidget(self.linear_slider, 3, 1)
        endeff_layout.addWidget(self.linear_lcd, 3, 2)

        control_layout.addWidget(self.endeff_group, alignment=Qt.AlignmentFlag.AlignTop)
        
        # ------------------ Left Bottom: Button Groups ------------------
        button_layout = QVBoxLayout()
        left_layout.addLayout(button_layout, stretch=1)
        
        # Up row: Horizontal layout for Motor Status and Target Detection.
        up_row_layout = QHBoxLayout()
        # Motor Status Group
        motor_status_group = QGroupBox("Motor Status")
        motor_status_group.setMaximumWidth(320)
        motor_status_group.setFont(font_label)
        motor_status_group.setStyleSheet("QGroupBox * { font-size: 14px; font-weight: normal; }")
        motor_status_layout = QVBoxLayout()
        motor_status_group.setLayout(motor_status_layout)

        # Motor homing button
        self.motor_home_btn = QPushButton("Motor Homing")
        self.motor_home_btn.setMaximumHeight(50)
        self.motor_home_btn.clicked.connect(self.motor_home)

        # Move to target posiont layout
        motor_move_layout = QVBoxLayout()
        self.motor_move_btn = QPushButton("Motor Move")
        self.motor_move_btn.setMaximumHeight(50)
        self.motor_move_btn.clicked.connect(self.motor_move)

        gantry_move_pos_layout = QHBoxLayout()
        gantry_move_pos_label = QLabel("Gantry Target\n(X,Y,Z): ")
        gantry_move_pos_label.setFixedWidth(100)
        self.gantry_move_x = QLineEdit()
        gantry_move_x_validator = QDoubleValidator(0, 100, 2)
        self.gantry_move_x.setValidator(gantry_move_x_validator)
        self.gantry_move_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gantry_move_x.setText("{}".format(self.gantry_pos_des_x))
        self.gantry_move_y = QLineEdit()
        gantry_move_y_validator = QDoubleValidator(0, 100, 2)
        self.gantry_move_y.setValidator(gantry_move_y_validator)
        self.gantry_move_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gantry_move_y.setText("{}".format(self.gantry_pos_des_y))
        self.gantry_move_z = QLineEdit()
        gantry_move_z_validator = QDoubleValidator(0, 100, 2)
        self.gantry_move_z.setValidator(gantry_move_z_validator)
        self.gantry_move_z.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gantry_move_z.setText("{}".format(self.gantry_pos_des_z))

        ee_move_pos_layout = QHBoxLayout()

        ee_move_pos_label = QLabel("EE Target\n(servo, linear): ")
        ee_move_pos_label.setFixedWidth(100)
        self.ee_move_servo = QLineEdit()
        ee_move_servo_validator = QDoubleValidator(0, 180, 2)
        self.ee_move_servo.setValidator(ee_move_servo_validator)
        self.ee_move_servo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ee_move_servo.setText("{}".format(self.endeff_pos_des_servo))
        self.ee_move_linear = QLineEdit()
        ee_move_linear_validator = QDoubleValidator(0, 180, 2)
        self.ee_move_linear.setValidator(ee_move_linear_validator)
        self.ee_move_linear.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ee_move_linear.setText("{}".format(self.endeff_pos_des_linear))

        gantry_move_pos_layout.addWidget(gantry_move_pos_label)
        gantry_move_pos_layout.addWidget(self.gantry_move_x)
        gantry_move_pos_layout.addWidget(self.gantry_move_y)
        gantry_move_pos_layout.addWidget(self.gantry_move_z)

        ee_move_pos_layout.addWidget(ee_move_pos_label)
        ee_move_pos_layout.addWidget(self.ee_move_servo)
        ee_move_pos_layout.addWidget(self.ee_move_linear)

        motor_move_layout.addWidget(self.motor_move_btn)
        motor_move_layout.addLayout(gantry_move_pos_layout)
        motor_move_layout.addLayout(ee_move_pos_layout)
        
        motor_status_label_layout = QHBoxLayout()
        fixed_status_label = QLabel("Status:")
        fixed_status_label.setFixedWidth(80)
        self.motor_status = QLabel("Not Homed")
        motor_status_label_layout.addWidget(fixed_status_label)
        motor_status_label_layout.addWidget(self.motor_status)

        motor_position_layout = QVBoxLayout()
        gantry_pos_layout = QHBoxLayout()
        gantry_pos_label = QLabel("Gantry\nPosition:")
        gantry_pos_label.setFixedWidth(100)
        self.gantry_pos_x = QLabel("N/A")
        self.gantry_pos_x.setFixedWidth(50)
        self.gantry_pos_y = QLabel("N/A")
        self.gantry_pos_y.setFixedWidth(50)
        self.gantry_pos_z = QLabel("N/A")
        self.gantry_pos_z.setFixedWidth(50)
        gantry_pos_layout.addWidget(gantry_pos_label)
        gantry_pos_layout.addWidget(self.gantry_pos_x)
        gantry_pos_layout.addWidget(self.gantry_pos_y)
        gantry_pos_layout.addWidget(self.gantry_pos_z)

        ee_pos_layout = QHBoxLayout()
        ee_pos_label = QLabel("EE Position:")
        ee_pos_label.setFixedWidth(100)
        self.ee_pos_servo = QLabel("N/A")
        self.ee_pos_servo.setFixedWidth(50)
        self.ee_pos_linear = QLabel("N/A")
        self.ee_pos_linear.setFixedWidth(50)
        ee_pos_layout.addWidget(ee_pos_label)
        ee_pos_layout.addWidget(self.ee_pos_servo)
        ee_pos_layout.addWidget(self.ee_pos_linear)

        motor_position_layout.addLayout(gantry_pos_layout)
        motor_position_layout.addLayout(ee_pos_layout)

        motor_status_layout.addWidget(self.motor_home_btn)
        motor_status_layout.addLayout(motor_move_layout)
        motor_status_layout.addLayout(motor_status_label_layout)
        motor_status_layout.addLayout(motor_position_layout)

        up_row_layout.addWidget(motor_status_group)
        
        # Target Detection Group
        detection_group = QGroupBox("Target Detection")
        detection_group.setFont(font_label)
        detection_group.setStyleSheet("QGroupBox * { font-size: 14px; font-weight: normal; }")
        detection_layout = QVBoxLayout()
        detection_group.setLayout(detection_layout)

        self.detection_btn = QPushButton("Start Target Detection")
        self.detection_btn.setMaximumHeight(50)
        self.detection_btn.clicked.connect(self.start_detection)
        detection_layout.addWidget(self.detection_btn)

        cam1_status_layout = QHBoxLayout()
        cam1_fixed_label = QLabel("Camera 1:")
        self.cam1_detection_status = QLabel("Not Detected")
        cam1_status_layout.addWidget(cam1_fixed_label)
        cam1_status_layout.addWidget(self.cam1_detection_status)
        detection_layout.addLayout(cam1_status_layout)

        cam2_status_layout = QHBoxLayout()
        cam2_fixed_label = QLabel("Camera 2:")
        self.cam2_detection_status = QLabel("Not Detected")
        cam2_status_layout.addWidget(cam2_fixed_label)
        cam2_status_layout.addWidget(self.cam2_detection_status)
        detection_layout.addLayout(cam2_status_layout)

        up_row_layout.addWidget(detection_group)

        # Down row: Horizontal layout for Stages: Localization and Liver Biopsy
        down_row_layout = QHBoxLayout()
        # Stage 1 Group: Localization (2-step functions)
        self.localization_group = QGroupBox("Stage 1: Localization")
        self.localization_group.setFont(font_label)
        self.localization_group.setStyleSheet("QGroupBox * { font-size: 14px; font-weight: normal; }")
        localization_layout = QVBoxLayout()
        self.localization_group.setLayout(localization_layout)

        positioning_row1_layout = QHBoxLayout()
        positioning_row1_fixed_label = QLabel("X: ")
        positioning_row1_fixed_label.setFixedWidth(40)
        positioning_row1_layout.addWidget(positioning_row1_fixed_label)
        self.positioning_row1_btn = QPushButton("Find X")
        self.positioning_row1_btn.setMaximumHeight(50)
        self.positioning_row1_btn.clicked.connect(self.positioning_X)
        positioning_row1_layout.addWidget(self.positioning_row1_btn)
        self.positioning_row1_status = QLabel("Not Positioned")
        positioning_row1_layout.addWidget(self.positioning_row1_status)
        localization_layout.addLayout(positioning_row1_layout)

        positioning_row2_layout = QHBoxLayout()
        positioning_row2_fixed_label = QLabel("Y&Z: ")
        positioning_row2_fixed_label.setFixedWidth(40)
        positioning_row2_layout.addWidget(positioning_row2_fixed_label)
        self.positioning_row2_btn = QPushButton("Find Y&Z")
        self.positioning_row2_btn.setMaximumHeight(50)
        self.positioning_row2_btn.clicked.connect(self.positioning_YZ)
        positioning_row2_layout.addWidget(self.positioning_row2_btn)
        self.positioning_row2_status = QLabel("Not Positioned")
        positioning_row2_layout.addWidget(self.positioning_row2_status)
        localization_layout.addLayout(positioning_row2_layout)

        down_row_layout.addWidget(self.localization_group)

        # Stage 2 Group: Liver Boipsy (4-step status)
        self.liver_boipsy_group = QGroupBox("Stage 2: Liver Boipsy")
        self.liver_boipsy_group.setFont(font_label)
        self.liver_boipsy_group.setStyleSheet("QGroupBox * { font-size: 14px; font-weight: normal; }")
        liver_boipsy_layout = QVBoxLayout()
        self.liver_boipsy_group.setLayout(liver_boipsy_layout)

        self.liver_boipsy_btn = QPushButton("Liver Boipsy")
        self.liver_boipsy_btn.setMaximumHeight(50)
        self.liver_boipsy_btn.clicked.connect(self.start_liver_boipsy)
        liver_boipsy_layout.addWidget(self.liver_boipsy_btn)

        self.liver_status_labels = []
        for step in range(1, 5):
            step_layout = QHBoxLayout()
            fixed_label = QLabel(f"Step {step}:")
            dynamic_label = QLabel("Wait")
            step_layout.addWidget(fixed_label)
            step_layout.addWidget(dynamic_label)
            self.liver_status_labels.append(dynamic_label)
            liver_boipsy_layout.addLayout(step_layout)

        down_row_layout.addWidget(self.liver_boipsy_group)

        button_layout.addLayout(up_row_layout)
        button_layout.addLayout(down_row_layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_views)
        self.cameras_active = False
        
        self.cam1 = Camera(0, "Camera 1", use_csi=True, sensor_id=0)
        self.cam2 = Camera(1, "Camera 2", use_csi=True, sensor_id=1)


        self.arduino1 = None
        self.arduino1_rate = 115200
        self.serial_thread1 = None
        self.arduino2 = None
        self.arduino2_rate = 115200
        self.serial_thread2 = None
        self.sender_threads = []

        self.gantry_gui_enabled = False
        self.endeff_gui_enabled = False
        
        self.colormask_cam1 = False
        self.target_finded_cam1 = False
        self.colormask_cam2 = False
        self.target_finded_cam2 = False

    def toggle_cameras(self, checked):
        if checked:
            self.cam1.start_cap()
            self.cam2.start_cap()
            
            self.cam_toggle_btn.setText("Close Cameras")
            self.cameras_active = True
            self.timer.start(30)
        else:
            self.cam_toggle_btn.setText("Open Cameras")
            self.cameras_active = False
            self.timer.stop()
            self.cam_label1.clear()
            self.cam_label2.clear()

            self.cam1.release()
            self.cam2.release()

    def update_camera_views(self):
        if self.cameras_active:
            ret1, frame1 = self.cam1.read_frame()
            ret2, frame2 = self.cam2.read_frame()

            if ret1:
                if self.colormask_cam1:
                    self.cam1.detect_target()
                frame1 = self.cam1.get_display_frame()
                if frame1 is not None:
                    h, w, ch = frame1.shape
                    bytes_per_line = ch * w
                    image1 = QImage(frame1.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.cam_label1.setPixmap(QPixmap.fromImage(image1).scaled(self.cam_label1.size(), Qt.AspectRatioMode.KeepAspectRatio))

            if ret2:
                if self.colormask_cam2:
                    self.cam2.detect_target()
                frame2 = self.cam2.get_display_frame()
                if frame2 is not None:
                    h, w, ch = frame2.shape
                    bytes_per_line = ch * w
                    image2 = QImage(frame2.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.cam_label2.setPixmap(QPixmap.fromImage(image2).scaled(self.cam_label2.size(), Qt.AspectRatioMode.KeepAspectRatio))

            if self.cam1.target_found and self.cam1.last_target_center:
                c1x, c1y = self.cam1.last_target_center
                dis1_x, dis1_y = self.cam1.calculate_center_distance(c1x, c1y)
                self.cam1_detection_status.setText(f"Target at ({c1x}, {c1y})\nDistance to the camera center: ({dis1_x}, {dis1_y})")
                self.cam1_detection_status.setStyleSheet("color: green;")
            else:
                self.cam1_detection_status.setText("No Target")
                self.cam1_detection_status.setStyleSheet("color: red;")

            if self.cam2.target_found and self.cam2.last_target_center:
                c2x, c2y = self.cam2.last_target_center
                target_3D = self.cam2.calculate_world_3D(c2x, c2y)
                self.cam2_detection_status.setText("Target at ({}, {})\n3D Coordinate of the world frame: ({:.2f}, {:.2f}, {:.2f})".format(c2x,c2y,target_3D[0],target_3D[1],target_3D[2]))
                self.cam2_detection_status.setStyleSheet("color: green;")
            else:
                self.cam2_detection_status.setText("No Target")
                self.cam2_detection_status.setStyleSheet("color: red;")

    def parse_gantry_position(line):
        m = re.search(r"^Current Position:\s*X([\d\.-]+)\s+Y([\d\.-]+)\s+Z([\d\.-]+)", line)
        if m:
            posX = float(m.group(1))
            posY = float(m.group(2))
            posZ = float(m.group(3))
            return posX, posY, posZ
        return None

    def handle_arduino1_data(self, data):
        print("Arduino 1:", data)
        pos = self.parse_gantry_position(data)
        if pos is not None:
            self.gantry_X, self.gantry_Y, self.gantry_Z = pos
            self.gantry_pos_x.setText(f"{self.gantry_X:.2f}")
            self.gantry_pos_y.setText(f"{self.gantry_Y:.2f}")
            self.gantry_pos_z.setText(f"{self.gantry_Z:.2f}")

    def handle_arduino2_data(self, data):
        print("Arduino 2:", data)

    def send_command_in_thread_once(self, arduino, command):
        sender = SerialSenderThread(arduino, command)
        sender.finished.connect(lambda: self.sender_threads.remove(sender))
        self.sender_threads.append(sender)
        sender.start()

    def update_gantry(self, idx, value):
        if self.gantry_gui_enabled:
            print(f"Gantry Stepper {idx+1} set to {value}")
        else:
            print(f"GUI disabled; ignoring Stepper {idx+1}")

    def update_endeff(self, component, value):
        if self.endeff_gui_enabled:
            print(f"End-Effector {component} set to {value}")
        else:
            print(f"GUI disabled; ignoring {component}")

    def toggle_gantry_gui(self, checked):
        self.gantry_gui_enabled = checked
        if checked:
            self.gantry_gui_toggle.setText("Stop Sending Positional Command")
            if self.gantry_port_combo.currentText() != "Waiting":
                self.arduino1 = serial.Serial(self.gantry_port_combo.currentText(), self.arduino1_rate)
                print("Gantry port opened at:", self.gantry_port_combo.currentText())
                self.serial_thread1 = SerialReaderThread(self.arduino1)
                self.serial_thread1.data_received.connect(self.handle_arduino1_data)
                self.serial_thread1.start()
        else:
            self.gantry_gui_toggle.setText("Send Positional Command")
            if self.arduino1 is not None:
                print("Gantry port closed")
                self.serial_thread1.stop()
                self.arduino1.close()

    def toggle_endeff_gui(self, checked):
        self.endeff_gui_enabled = checked
        if checked:
            self.endeff_gui_toggle.setText("Stop Sending Positional Command")
            if self.endeff_port_combo.currentText() != "Waiting":
                self.arduino2 = serial.Serial(self.endeff_port_combo.currentText(), self.arduino2_rate)
                print("End-Effector port opened at:", self.endeff_port_combo.currentText())
                self.serial_thread2 = SerialReaderThread(self.arduino2)
                self.serial_thread2.data_received.connect(self.handle_arduino2_data)
                self.serial_thread2.start()
        else:
            self.endeff_gui_toggle.setText("Send Positional Command")
            if self.arduino2 is not None:
                print("End-Effector port closed")
                self.serial_thread2.stop()
                self.arduino2.close()

    def update_gantry_ports(self):
        ports = list_ports.comports()
        self.gantry_port_combo.clear()
        port_names = [port.device for port in ports]
        if not port_names:
            port_names = ["No ports available"]
        self.gantry_port_combo.addItems(port_names)
        print("Gantry ports updated:", port_names)

    def update_endeff_ports(self):
        ports = list_ports.comports()
        self.endeff_port_combo.clear()
        port_names = [port.device for port in ports]
        if not port_names:
            port_names = ["No ports available"]
        self.endeff_port_combo.addItems(port_names)
        print("End-Effector ports updated:", port_names)

    def motor_home(self):
        print("Executing motor homing")
        self.motor_status.setText("Homing...")
        QTimer.singleShot(2000, lambda: self.motor_status.setText("Homed"))

    def motor_move(self):
        self.motor_status.setText("Moving...")
        self.gantry_pos_des_x = float(self.gantry_move_x.text())
        self.gantry_pos_des_y = float(self.gantry_move_y.text())
        self.gantry_pos_des_z = float(self.gantry_move_z.text())
        self.endeff_pos_des_servo = float(self.ee_move_servo.text())
        self.endeff_pos_des_linear = float(self.ee_move_linear.text())

        print(f"Moving to: X{self.gantry_pos_des_x} Y{self.gantry_pos_des_y} Z{self.gantry_pos_des_z} Servo{self.endeff_pos_des_servo} Linear{self.endeff_pos_des_linear}")
        if self.arduino1 is not None:
            command_1 = f"Gantry move to: X{self.gantry_pos_des_x} Y{self.gantry_pos_des_y} Z{self.gantry_pos_des_z}"
            self.send_command_in_thread_once(self.arduino1, command_1)
        if self.arduino2 is not None:
            command_2 = f"Endeffector move to: Servo{self.endeff_pos_des_servo} Linear{self.endeff_pos_des_linear}"
            self.send_command_in_thread_once(self.arduino2, command_2)
        QTimer.singleShot(2000, lambda: self.motor_status.setText("Moved"))

    def start_detection(self):
        print("Starting target detection")
        self.colormask_cam1 = True
        self.colormask_cam2 = True
        def update_detection_status():
            if self.cam1.target_found:
                self.cam1_detection_status.setText("Target Detected")
                self.cam1_detection_status.setStyleSheet("color: green;")
            else:
                self.cam1_detection_status.setText("No Target")
                self.cam1_detection_status.setStyleSheet("color: red;")

            if self.cam2.target_found:
                self.cam2_detection_status.setText("Target Detected")
                self.cam2_detection_status.setStyleSheet("color: green;")
            else:
                self.cam2_detection_status.setText("No Target")
                self.cam2_detection_status.setStyleSheet("color: red;")
        QTimer.singleShot(200, update_detection_status)

    def positioning_X(self):
        print("Positioning X")
        self.positioning_row1_status.setText("Pixel Error: 0\nGantry Position: 0")

    def positioning_YZ(self):
        print("Positioning YZ")
        self.positioning_row2_status.setText("Target Y: 0\nTarget Z: 0\nRibcage Y: 0\nRibcage Z: 0")

    def start_liver_boipsy(self):
        print("Starting Liver Boipsy")
        for lbl in self.liver_status_labels:
            lbl.setText("Wait")
        QTimer.singleShot(1000, lambda: (self.liver_status_labels[0].setText("Pass"), self.liver_status_labels[0].setStyleSheet("color: green;")))
        QTimer.singleShot(2000, lambda: (self.liver_status_labels[1].setText("Pass"), self.liver_status_labels[1].setStyleSheet("color: green;")))
        QTimer.singleShot(3000, lambda: (self.liver_status_labels[2].setText("Pass"), self.liver_status_labels[2].setStyleSheet("color: green;")))
        QTimer.singleShot(4000, lambda: (self.liver_status_labels[3].setText("Pass"), self.liver_status_labels[3].setStyleSheet("color: green;")))

    def closeEvent(self, event):
        self.cam1.release()
        self.cam2.release()
        if self.arduino1 is not None:
            self.serial_thread1.stop()
            self.arduino1.close()
        if self.arduino2 is not None:
            self.serial_thread2.stop()
            self.arduino2.close()
        event.accept()

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotControlWindow()
    window.show()
    sys.exit(app.exec())
