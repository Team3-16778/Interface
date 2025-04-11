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
from utils import Gantry, Camera, ConnectionWorker, EndEffector



class InterfaceLite(QMainWindow):
    def __init__(self):
        super().__init__()


        ## ------------------ Inner Parameters ------------------

        self.last_connected_port = None
        self.last_ee_connected_port = None

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
        # self.cam_toggle_btn.toggled.connect(self.toggle_cameras) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        camera_controls_layout.addWidget(self.cam_toggle_btn, stretch=2)

        self.cam1_tune_btn = QPushButton("Color Mask 1")
        self.cam1_tune_btn.setMaximumHeight(50)
        self.cam1_tune_btn.setFont(font_button_1)
        self.cam1_tune_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        # self.cam1_tune_btn.clicked.connect(lambda: self.cam1.open_tuner()) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        camera_controls_layout.addWidget(self.cam1_tune_btn, stretch=1)

        self.cam2_tune_btn = QPushButton("Color Mask 2")
        self.cam2_tune_btn.setMaximumHeight(50)
        self.cam2_tune_btn.setFont(font_button_1)
        self.cam2_tune_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        # self.cam2_tune_btn.clicked.connect(lambda: self.cam2.open_tuner()) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        camera_controls_layout.addWidget(self.cam2_tune_btn, stretch=1)

        right_layout.addLayout(camera_controls_layout)

                # ------------------ Left Top: Control Panels ------------------
        control_layout = QHBoxLayout()
        left_layout.addLayout(control_layout)

        # Gantry Control Panel (existing code)
        self.gantry_group = QGroupBox("Gantry Control")
        self.gantry_group.setFont(font_label)
        self.gantry_group.setStyleSheet("QGroupBox * { font-size: 12px; font-weight: normal; }")
        gantry_layout = QGridLayout()
        self.gantry_group.setLayout(gantry_layout)

        # Initialize GantryController
        self.gantry = Gantry()
        self.gantry.position_updated.connect(self.update_gantry_position_display)
        self.gantry.target_updated.connect(self.update_gantry_target_display)

        # USB port selection and refresh button
        gantry_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.gantry_port_combo = QComboBox()
        self.gantry_port_combo.addItems(["Waiting"])
        gantry_layout.addWidget(self.gantry_port_combo, 0, 1)
        self.gantry_refresh_btn = QPushButton("Refresh\nPorts")
        self.gantry_refresh_btn.clicked.connect(self.update_gantry_ports)
        gantry_layout.addWidget(self.gantry_refresh_btn, 0, 2)

        self.connection_status = QLabel("Disconnected")
        self.connection_status.setAlignment(Qt.AlignCenter)
        gantry_layout.addWidget(self.connection_status, 1, 0, 1, 3) 

        self.gantry_port_combo.currentTextChanged.connect(self.handle_gantry_port_change)

        # Add sliders and LCD displays for three steppers
        self.gantry_sliders = []
        self.gantry_lcds = []
        labels = ["Gantry X", "Gantry Y", "Gantry Z"]

        for i in range(3):
            row = i + 2
            
            # Label
            label = QLabel(labels[i])
            label.setFixedWidth(60)
            
            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setMinimumWidth(150)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            slider.valueChanged.connect(lambda value, idx=i: self.on_gantry_slider_change(idx, value))
            self.gantry_sliders.append(slider)
            
            # LCD Display
            lcd = QLCDNumber()
            lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
            lcd.display(50)
            lcd.setFixedWidth(60)
            self.gantry_lcds.append(lcd)
            
            # Connect slider to LCD
            slider.valueChanged.connect(lcd.display)
            
            # Add to layout
            gantry_layout.addWidget(label, row, 0)
            gantry_layout.addWidget(slider, row, 1)
            gantry_layout.addWidget(lcd, row, 2)

        gantry_layout.setColumnStretch(0, 0)
        gantry_layout.setColumnStretch(1, 1)
        gantry_layout.setColumnStretch(2, 0)

        control_layout.addWidget(self.gantry_group, alignment=Qt.AlignmentFlag.AlignTop)

        # ------------------ End Effector Control Panel ------------------
        self.ee_group = QGroupBox("End Effector Control")
        self.ee_group.setFont(font_label)
        self.ee_group.setStyleSheet("QGroupBox * { font-size: 12px; font-weight: normal; }")
        ee_layout = QGridLayout()
        self.ee_group.setLayout(ee_layout)

        # Initialize EndEffector
        self.end_effector = EndEffector()
        self.end_effector.position_updated.connect(self.update_ee_position_display)
        self.end_effector.target_updated.connect(self.update_ee_target_display)

        # USB port selection and refresh button
        ee_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.ee_port_combo = QComboBox()
        self.ee_port_combo.addItems(["Waiting"])
        ee_layout.addWidget(self.ee_port_combo, 0, 1)
        self.ee_refresh_btn = QPushButton("Refresh\nPorts")
        self.ee_refresh_btn.clicked.connect(self.update_ee_ports)
        ee_layout.addWidget(self.ee_refresh_btn, 0, 2)

        self.ee_connection_status = QLabel("Disconnected")
        self.ee_connection_status.setAlignment(Qt.AlignCenter)
        ee_layout.addWidget(self.ee_connection_status, 1, 0, 1, 3)

        self.ee_port_combo.currentTextChanged.connect(self.handle_ee_port_change)

        # Add sliders and LCD displays for theta and x
        self.ee_sliders = []
        self.ee_lcds = []
        ee_labels = ["Rotation (θ)", "Linear (X)"]
        ee_ranges = [(0, 180), (0, 100)]  # Theta: 0-180 degrees, X: 0-100 mm

        for i in range(2):
            row = i + 2
            
            # Label
            label = QLabel(ee_labels[i])
            label.setFixedWidth(80)
            
            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(ee_ranges[i][0], ee_ranges[i][1])
            slider.setValue(ee_ranges[i][1] // 2)  # Start at middle position
            slider.setMinimumWidth(150)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            slider.valueChanged.connect(lambda value, idx=i: self.on_ee_slider_change(idx, value))
            self.ee_sliders.append(slider)
            
            # LCD Display
            lcd = QLCDNumber()
            lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
            lcd.display(ee_ranges[i][1] // 2)
            lcd.setFixedWidth(80)
            self.ee_lcds.append(lcd)
            
            # Connect slider to LCD
            slider.valueChanged.connect(lcd.display)
            
            # Add to layout
            ee_layout.addWidget(label, row, 0)
            ee_layout.addWidget(slider, row, 1)
            ee_layout.addWidget(lcd, row, 2)

        # Homing button
        self.ee_home_btn = QPushButton("Home End Effector")
        self.ee_home_btn.clicked.connect(self.end_effector.home)
        ee_layout.addWidget(self.ee_home_btn, 4, 0, 1, 3)

        ee_layout.setColumnStretch(0, 0)
        ee_layout.setColumnStretch(1, 1)
        ee_layout.setColumnStretch(2, 0)

        control_layout.addWidget(self.ee_group, alignment=Qt.AlignmentFlag.AlignTop)

        # Update ports for both gantry and end effector
        self.update_gantry_ports()
        self.update_ee_ports()


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
        # self.motor_home_btn.clicked.connect(self.motor_home) <<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Move to target posiont layout
        motor_move_layout = QVBoxLayout()
        self.motor_move_btn = QPushButton("Motor Move")
        self.motor_move_btn.setMaximumHeight(50)
        # self.motor_move_btn.clicked.connect(self.motor_move) <<<<<<<<<<<<<<<<<<<<<<<<<<<<

        gantry_move_pos_layout = QHBoxLayout()
        gantry_move_pos_label = QLabel("Gantry Target\n(X,Y,Z): ")
        gantry_move_pos_label.setFixedWidth(100)
        self.gantry_move_x = QLineEdit()
        gantry_move_x_validator = QDoubleValidator(0, 100, 2)
        self.gantry_move_x.setValidator(gantry_move_x_validator)
        self.gantry_move_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.gantry_move_x.setText("{}".format(self.gantry_pos_des_x)) <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.gantry_move_y = QLineEdit()
        gantry_move_y_validator = QDoubleValidator(0, 100, 2)
        self.gantry_move_y.setValidator(gantry_move_y_validator)
        self.gantry_move_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.gantry_move_y.setText("{}".format(self.gantry_pos_des_y)) <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.gantry_move_z = QLineEdit()
        gantry_move_z_validator = QDoubleValidator(0, 100, 2)
        self.gantry_move_z.setValidator(gantry_move_z_validator)
        self.gantry_move_z.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.gantry_move_z.setText("{}".format(self.gantry_pos_des_z)) <<<<<<<<<<<<<<<<<<<<<<<<<<<<

        ee_move_pos_layout = QHBoxLayout()

        ee_move_pos_label = QLabel("EE Target\n(servo, linear): ")
        ee_move_pos_label.setFixedWidth(100)
        self.ee_move_servo = QLineEdit()
        ee_move_servo_validator = QDoubleValidator(0, 180, 2)
        self.ee_move_servo.setValidator(ee_move_servo_validator)
        self.ee_move_servo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.ee_move_servo.setText("{}".format(self.endeff_pos_des_servo)) <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.ee_move_linear = QLineEdit()
        ee_move_linear_validator = QDoubleValidator(0, 180, 2)
        self.ee_move_linear.setValidator(ee_move_linear_validator)
        self.ee_move_linear.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.ee_move_linear.setText("{}".format(self.endeff_pos_des_linear)) <<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
        # self.detection_btn.clicked.connect(self.start_detection) <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        # self.positioning_row1_btn.clicked.connect(self.positioning_X) <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        # self.positioning_row2_btn.clicked.connect(self.positioning_YZ) <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        # self.liver_boipsy_btn.clicked.connect(self.start_liver_boipsy) <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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


    ##### GANTRY GUI CONTROLS #####
    
    def update_gantry_ports(self):
        """Refresh available serial ports without triggering errors"""
        # Store current selection
        current_port = self.gantry_port_combo.currentText()
        
        # Block signals during update
        self.gantry_port_combo.blockSignals(True)
        
        # Clear and repopulate
        self.gantry_port_combo.clear()
        ports = [port.device for port in list_ports.comports()]
        
        if not ports:
            self.gantry_port_combo.addItem("No ports available")
        else:
            self.gantry_port_combo.addItems(ports)
            # Restore previous selection if still available
            if current_port in ports:
                self.gantry_port_combo.setCurrentText(current_port)
        
        # Restore signal handling
        self.gantry_port_combo.blockSignals(False)
        
        print("Ports updated:", ports or "No ports found")

    def toggle_gantry_control(self, enabled):
        """Handle gantry control toggle"""
        self.gantry_gui_enabled = enabled
        
        if enabled:
            self.gantry_gui_toggle.setText("Stop Sending Positional Command")
            port = self.gantry_port_combo.currentText()
            if port not in ["Waiting", "No ports available"]:
                if not self.gantry.connect(port):
                    self.gantry_gui_toggle.setChecked(False)
                    self.gantry_gui_enabled = False
        else:
            self.gantry_gui_toggle.setText("Send Positional Command")
            self.gantry.disconnect()

    def on_gantry_slider_change(self, axis, value):
        """Handle gantry slider changes"""
        if not self.gantry.is_connected:
            return
        
        # Get current target from controller
        current_target = list(self.gantry.target_position)
        
        # Update the appropriate axis
        current_target[axis] = float(value)
        
        # Update through controller (will emit target_updated signal)
        self.gantry.target_position = tuple(current_target)
        
        # Send move command immediately
        self.gantry.move_to(*current_target)

    def update_gantry_position_display(self, x, y, z):
        """Update UI with current gantry position"""
        self.gantry_pos_x.setText(f"{x:.2f}")
        self.gantry_pos_y.setText(f"{y:.2f}")
        self.gantry_pos_z.setText(f"{z:.2f}")

    def update_gantry_target_display(self, x, y, z):
        """Update UI with target gantry position"""
        # Update sliders without triggering valueChanged signals
        for i, slider in enumerate(self.gantry_sliders):
            slider.blockSignals(True)
            slider.setValue(int([x, y, z][i]))
            slider.blockSignals(False)
        
        # Update LCD displays
        for i, lcd in enumerate(self.gantry_lcds):
            lcd.display(int([x, y, z][i]))

    def handle_gantry_port_change(self, port):
        """Handle port selection without freezing GUI"""
        # Ignore placeholder texts
        if not port or port in ["No ports available"]:
            self.update_connection_status("Disconnected", "red")
            return
        
        # Show connecting status immediately
        self.update_connection_status(f"Connecting to {port}...", "orange")
        
        # Setup worker thread
        self.connection_thread = QThread()
        self.connection_worker = ConnectionWorker(self.gantry, port)
        self.connection_worker.moveToThread(self.connection_thread)
        
        # Connect signals
        self.connection_thread.started.connect(self.connection_worker.run)
        self.connection_worker.finished.connect(self.handle_connection_result)
        self.connection_worker.finished.connect(self.connection_thread.quit)
        self.connection_worker.finished.connect(self.connection_worker.deleteLater)
        self.connection_thread.finished.connect(self.connection_thread.deleteLater)
        
        # Start the thread
        self.connection_thread.start()

    def update_connection_status(self, text, color):
        """Update connection status label"""
        self.connection_status.setText(text)
        self.connection_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def handle_connection_result(self, success):
        """Handle the final connection result"""
        if success:
            self.update_connection_status(f"Connected to {self.gantry.port}", "green")
        else:
            self.update_connection_status("Connection failed", "red")
            # Revert to previous port if available
            if hasattr(self, 'last_connected_port'):
                self.gantry_port_combo.setCurrentText(self.last_connected_port)
                
    ### END EFFECTOR GUI CONTROLS ###
    
    def update_ee_ports(self):
        """Refresh available serial ports for end effector"""
        current_port = self.ee_port_combo.currentText()
        
        self.ee_port_combo.blockSignals(True)
        self.ee_port_combo.clear()
        ports = [port.device for port in list_ports.comports()]
        
        if not ports:
            self.ee_port_combo.addItem("No ports available")
        else:
            self.ee_port_combo.addItems(ports)
            if current_port in ports:
                self.ee_port_combo.setCurrentText(current_port)
        
        self.ee_port_combo.blockSignals(False)

    def on_ee_slider_change(self, axis, value):
        """Handle end effector slider changes"""
        if not self.end_effector.is_connected:
            return
        
        # Get current target from controller
        current_target = list(self.end_effector.target_position)
        
        # Update the appropriate axis
        current_target[axis] = float(value)
        
        # Update through controller (will emit target_updated signal)
        self.end_effector.target_position = tuple(current_target)
        
        # Send move command immediately
        self.end_effector.move_to(*current_target)

    def update_ee_position_display(self, theta, x):
        """Update UI with current end effector position"""
        self.ee_pos_servo.setText(f"{theta:.2f}")
        self.ee_pos_linear.setText(f"{x:.2f}")

    def update_ee_target_display(self, theta, x):
        """Update UI with target end effector position"""
        # Update sliders without triggering valueChanged signals
        for i, slider in enumerate(self.ee_sliders):
            slider.blockSignals(True)
            slider.setValue(int([theta, x][i]))
            slider.blockSignals(False)
        
        # Update LCD displays
        for i, lcd in enumerate(self.ee_lcds):
            lcd.display(int([theta, x][i]))

    def handle_ee_port_change(self, port):
        """Handle port selection for end effector without freezing GUI"""
        if not port or port in ["No ports available"]:
            self.update_ee_connection_status("Disconnected", "red")
            return
        
        self.update_ee_connection_status(f"Connecting to {port}...", "orange")
        
        # Setup worker thread
        self.ee_connection_thread = QThread()
        self.ee_connection_worker = ConnectionWorker(self.end_effector, port)
        self.ee_connection_worker.moveToThread(self.ee_connection_thread)
        
        # Connect signals
        self.ee_connection_thread.started.connect(self.ee_connection_worker.run)
        self.ee_connection_worker.finished.connect(self.handle_ee_connection_result)
        self.ee_connection_worker.finished.connect(self.ee_connection_thread.quit)
        self.ee_connection_worker.finished.connect(self.ee_connection_worker.deleteLater)
        self.ee_connection_thread.finished.connect(self.ee_connection_thread.deleteLater)
        
        # Start the thread
        self.ee_connection_thread.start()

    def update_ee_connection_status(self, text, color):
        """Update end effector connection status label"""
        self.ee_connection_status.setText(text)
        self.ee_connection_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def handle_ee_connection_result(self, success):
        """Handle the final connection result for end effector"""
        if success:
            self.update_ee_connection_status(f"Connected to {self.end_effector.port}", "green")
            self.last_ee_connected_port = self.end_effector.port
        else:
            self.update_ee_connection_status("Connection failed", "red")
            # Revert to previous port if available
            if self.last_ee_connected_port:
                self.ee_port_combo.setCurrentText(self.last_ee_connected_port)

    ##### WINDOW CLOSE HANDLER #####
    def closeEvent(self, event):
        """Clean up threads when closing the window"""
        # Stop any ongoing connection attempts
        if hasattr(self, 'connection_thread') and self.connection_thread.isRunning():
            self.connection_thread.quit()
            self.connection_thread.wait(500)  # Wait up to 500ms
        
        # Disconnect hardware
        self.gantry.disconnect()
        
        # Proceed with normal close
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InterfaceLite()
    window.show()
    sys.exit(app.exec())
