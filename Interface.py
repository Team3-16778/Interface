import sys
import re
import serial
from serial.tools import list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QLCDNumber, QSizePolicy, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QGuiApplication, QFont, QDoubleValidator
from camera_utils import CSI_Camera, gstreamer_pipeline, colormask, calculate_world_3D
import cv2
import numpy as np

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

        
        # ------------------ Right: Camera Toggle and Displays ------------------
        # Camera 1 Group
        cam1_group = QGroupBox("Camera 1")
        cam1_group.setFont(font_label)
        cam1_layout = QVBoxLayout()
        cam1_group.setLayout(cam1_layout)
        self.cam_label1 = QLabel()
        self.cam_label1.setFixedSize(800, 600)
        self.cam_label1.setStyleSheet("background-color: #000;")
        cam1_layout.addWidget(self.cam_label1)
        right_layout.addWidget(cam1_group)

        # Camera toggle button on the right side
        self.cam_toggle_btn = QPushButton("Open Cameras")
        self.cam_toggle_btn.setCheckable(True)
        self.cam_toggle_btn.setFont(font_button_1)
        self.cam_toggle_btn.toggled.connect(self.toggle_cameras)
        right_layout.addWidget(self.cam_toggle_btn)
        
        # Camera 2 Group
        cam2_group = QGroupBox("Camera 2")
        cam2_group.setFont(font_label)
        cam2_layout = QVBoxLayout()
        cam2_group.setLayout(cam2_layout)
        self.cam_label2 = QLabel()
        self.cam_label2.setFixedSize(800, 600)
        self.cam_label2.setStyleSheet("background-color: #000;")
        cam2_layout.addWidget(self.cam_label2)
        right_layout.addWidget(cam2_group)
        
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
        # GUI control toggle button
        self.gantry_gui_toggle = QPushButton("Enable GUI Control")
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
        # Set the column stretches so that column 1 (slider) expands,
        # while columns 0 (label) and 2 (LCD) remain fixed.
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
        # GUI control toggle button
        self.endeff_gui_toggle = QPushButton("Enable GUI Control")
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

        # Camera 1 status: fixed and dynamic labels side by side
        cam1_status_layout = QHBoxLayout()
        cam1_fixed_label = QLabel("Camera 1:")
        self.cam1_detection_status = QLabel("Not Detected")
        cam1_status_layout.addWidget(cam1_fixed_label)
        cam1_status_layout.addWidget(self.cam1_detection_status)
        detection_layout.addLayout(cam1_status_layout)

        # Camera 2 status: fixed and dynamic labels side by side
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

        # Create two rows for the 2 steps
        # Row 1: X positioning: A fixed label, a button(start function), and a dynamic label for status.
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

        # Row 2: Y&Z positioning: A fixed label, a button(start function), and a dynamic label for status.
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

        # Create four rows for the 4 steps; each row contains a fixed label and a dynamic label.
        self.liver_status_labels = []  # Store only the dynamic part for updating
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
        
        ## ------------------ Timer, Camera, Arduino and flags Initialization ------------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_views)
        self.cameras_active = False
        
        # # Initialize 2 WebCam cameras (using OpenCV for demonstration)
        # self.cam1 = cv2.VideoCapture(0)
        # self.cam2 = cv2.VideoCapture(1)
        # Initialize 2 CSI cameras
        self.cam1 = CSI_Camera()
        self.cam2 = CSI_Camera()
        
        # Initialize 2 Arduinos
        # arduino1 is for gantry, arduino2 is for endeff
        self.arduino1 = None
        self.arduino1_rate = 115200
        self.serial_thread1 = None
        self.arduino2 = None
        self.arduino2_rate = 115200
        self.serial_thread2 = None
        self.sender_threads = []

        # GUI control enable flags
        self.gantry_gui_enabled = False
        self.endeff_gui_enabled = False

        # colormask flags for 2 cameras
        self.colormask_cam1 = False
        self.target_finded_cam1 = False
        self.colormask_cam2 = False
        self.target_finded_cam2 = False
        

    def toggle_cameras(self, checked):
        if checked:
            self.cam_toggle_btn.setText("Close Cameras")
            self.cameras_active = True
            self.timer.start(30)
        else:
            self.cam_toggle_btn.setText("Open Cameras")
            self.cameras_active = False
            self.timer.stop()
            self.cam_label1.clear()
            self.cam_label2.clear()
            
    def update_camera_views(self):
        if self.cameras_active:
            # set parameters for 2 cameras
            cam1_params = gstreamer_pipeline(
                sensor_id=1,
                capture_width=3280,
                capture_height=2464,
                flip_method=0,
                framerate=21,
                )
            cam2_params = gstreamer_pipeline(
                sensor_id=0,
                capture_width=3280,
                capture_height=2464,
                flip_method=0,
                framerate=21,
                )
            # open, start, and read 2 cameras
            self.cam1.open(cam1_params)
            self.cam1.start()
            self.cam2.open(cam2_params)
            self.cam2.start()
            if self.cam1.video_capture.isOpened():
                ret1, frame1 = self.cam1.read()
            else:
                ret1 = False
            if self.cam2.video_capture.isOpened():
                ret2, frame2 = self.cap2.read()
            else:
                ret2 = False
            
            # # test on WebCam
            # ret1, frame1 = self.cam1.read()
            # ret2, frame2 = self.cam2.read()

            if ret1:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                if self.colormask_cam1:
                    _, _, frame1, self.target_finded_cam1 = colormask(frame1)
                h, w, ch = frame1.shape
                bytes_per_line = ch * w
                image1 = QImage(frame1.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.cam_label1.setPixmap(QPixmap.fromImage(image1).scaled(self.cam_label1.size(), Qt.AspectRatioMode.KeepAspectRatio))
            if ret2:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                if self.colormask_cam2:
                    _, _, image2, self.target_finded_cam2 = colormask(frame2)
                    frame2 = np.hstack((frame2, image2))
                h, w, ch = frame2.shape
                bytes_per_line = ch * w
                image2 = QImage(frame2.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.cam_label2.setPixmap(QPixmap.fromImage(image2).scaled(self.cam_label2.size(), Qt.AspectRatioMode.KeepAspectRatio))


    def parse_gantry_position(line):
        # Match the string format "Current Position: X10.00 Y20.00 Z30.00"
        m = re.search(r"^Current Position:\s*X([\d\.\-]+)\s+Y([\d\.\-]+)\s+Z([\d\.\-]+)", line)
        if m:
            posX = float(m.group(1))
            posY = float(m.group(2))
            posZ = float(m.group(3))
            return posX, posY, posZ
        return None

    def handle_arduino1_data(self, data):
        print("Arduino 1:", data)
        # read and update the gantry position
        pos = self.parse_gantry_position(data)
        if pos is not None:
            self.gantry_X, self.gantry_Y, self.gantry_Z = pos
            self.gantry_pos_x.setText(f"{self.gantry_X:.2f}")
            self.gantry_pos_y.setText(f"{self.gantry_Y:.2f}")
            self.gantry_pos_z.setText(f"{self.gantry_Z:.2f}")

    
    def handle_arduino2_data(self, data):
        print("Arduino 2:", data)
        # update

    def send_command_in_thread_once(self, arduino, command):
        sender = SerialSenderThread(arduino, command)
        # when the thread finishes, remove it from the list
        sender.finished.connect(lambda: self.sender_threads.remove(sender))
        self.sender_threads.append(sender)
        sender.start()


    def update_gantry(self, idx, value):
        if self.gantry_gui_enabled:
            print(f"Gantry Stepper {idx+1} set to {value}")
        else:
            print(f"Gantry GUI control disabled; ignoring change for Stepper {idx+1}")
    
    def update_endeff(self, component, value):
        if self.endeff_gui_enabled:
            print(f"End-Effector {component} set to {value}")
        else:
            print(f"End-Effector GUI control disabled; ignoring change for {component}")
    
    def toggle_gantry_gui(self, checked):
        self.gantry_gui_enabled = checked
        if checked:
            self.gantry_gui_toggle.setText("Disable GUI Control")
            if self.gantry_port_combo.currentText() != "Waiting":
                self.arduino1 = serial.Serial(self.gantry_port_combo.currentText(), self.arduino1_rate)
                print("Gantry port opened at: ", self.gantry_port_combo.currentText())
                self.serial_thread1 = SerialReaderThread(self.arduino1)
                self.serial_thread1.data_received.connect(self.handle_arduino1_data)
                self.serial_thread1.start()
        else:
            self.gantry_gui_toggle.setText("Enable GUI Control")
            if self.arduino1 is not None:
                print("Gantry port closed")
                self.serial_thread1.stop()
                self.arduino1.close()

    
    def toggle_endeff_gui(self, checked):
        self.endeff_gui_enabled = checked
        if checked:
            self.endeff_gui_toggle.setText("Disable GUI Control")
            if self.endeff_port_combo.currentText() != "Waiting":
                self.arduino2 = serial.Serial(self.endeff_port_combo.currentText(), self.arduino2_rate)
                print("End-Effector port opened at: ", self.endeff_port_combo.currentText())
                self.serial_thread2 = SerialReaderThread(self.arduino2)
                self.serial_thread2.data_received.connect(self.handle_arduino2_data)
                self.serial_thread2.start()
        else:
            self.endeff_gui_toggle.setText("Enable GUI Control")
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

        print("Executing motor movement to: X{} Y{} Z{} Servo{} Linear{}".format(self.gantry_pos_des_x, self.gantry_pos_des_y, self.gantry_pos_des_z, self.endeff_pos_des_servo, self.endeff_pos_des_linear))
        if self.arduino1 is not None:
            command_1 = "Gantry move to: X{} Y{} Z{}".format(self.gantry_pos_des_x, self.gantry_pos_des_y, self.gantry_pos_des_z)
            self.send_command_in_thread_once(self.arduino1, command_1)
        if self.arduino2 is not None:
            command_2 = "Endeffector move to: Servo{} Linear{}".format(self.endeff_pos_des_servo, self.endeff_pos_des_linear)
            self.send_command_in_thread_once(self.arduino2, command_2)
        QTimer.singleShot(2000, lambda: self.motor_status.setText("Moved"))
    
    def start_detection(self):
        print("Starting target detection process")
        self.cam1_detection_status.setText("Detecting...")
        self.colormask_cam1 = not self.colormask_cam1
        self.cam2_detection_status.setText("Detecting...")
        self.colormask_cam2 = not self.colormask_cam2
        QTimer.singleShot(3000, lambda: (
            self.cam1_detection_status.setText("Target Detected"),
            self.cam1_detection_status.setStyleSheet("color: green;")
        ))
        QTimer.singleShot(3000, lambda: (
            self.cam2_detection_status.setText("No Target"),
            self.cam2_detection_status.setStyleSheet("color: red;")
        ))

    def positioning_X(self):
        print("Starting X positioning process")
        self.positioning_row1_status.setText(" Pixel Error: {}\n Gantry Position: {}".format(0, 0))

    def positioning_YZ(self):
        print("Starting YZ positioning process")
        self.positioning_row2_status.setText(" Target Y: {}\n Target Z: {}\n Ribcage Y: {}\n Ribcage Z: {}".format(0, 0, 0, 0))

    def start_liver_boipsy(self):
        print("Starting Liver Boipsy process")
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
    data_received = pyqtSignal(str)  # define the signal
    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.running = True

    def run(self):
        while self.running:
            if self.serial_port.in_waiting:
                try:
                    # read a line from the serial port
                    data = self.serial_port.readline().decode('utf-8').strip()
                    if data:
                        self.data_received.emit(data)
                except Exception as e:
                    print("Error reading serial:", e)
            self.msleep(1)  # delay, reduces CPU usage

    def stop(self):
        self.running = False
        self.wait()  # Wait for the thread to finish

class SerialSenderThread(QThread):
    def __init__(self, arduino, command):
        super().__init__()
        self.arduino = arduino
        self.command = command

    def run(self):
        # write the command to the serial port
        if self.arduino.is_open:
            self.arduino.write((self.command + "\n").encode("utf-8"))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotControlWindow()
    window.show()
    sys.exit(app.exec())
