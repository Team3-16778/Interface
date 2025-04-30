import sys
import os
import serial
from serial.tools import list_ports
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QLCDNumber, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QPointF
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont, QDoubleValidator
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

import cv2
import numpy as np
from ColorMask import ColorMask
from utils import Gantry, EndEffector, Camera, CameraHandler, HardwareManager



class InterfaceLite(QMainWindow):
    def __init__(self, hardware_manager: HardwareManager = None):
        super().__init__()

        self.hw = hardware_manager
        self.gantry = self.hw.gantry if hardware_manager else None
        self.end_effector = self.hw.end_effector if hardware_manager else None  
        self.camera1 = self.hw.camera1 if hardware_manager else None
        self.camera2 = self.hw.camera2 if hardware_manager else None
        self.camera1.gui_active = True
        self.camera2.gui_active = True


        ## ------------------ Inner Parameters ------------------

        self.gantry_x = 50
        self.gantry_y = 50
        self.gantry_z = 50

        self.theta = 0
        self.delta = 0

        self.cam1_detection_status = None
        self.cam2_detection_status = None

        self.x_error = None

        self.target_y_history = []  # List to store target y positions
        self.max_history_length = 500  # Maximum number of points to store

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

         ## ------------------ Gatnry Control Panel ------------------

        control_layout = QHBoxLayout()
        left_layout.addLayout(control_layout)

        self.gantry_group = QGroupBox("Gantry Control")
        self.gantry_group.setFont(font_label)
        gantry_layout = QGridLayout()
        self.gantry_group.setLayout(gantry_layout)

        gantry_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.gantry_port_combo = QComboBox()
        self.gantry_port_combo.addItems(["Waiting"])
        gantry_layout.addWidget(self.gantry_port_combo, 0, 1)

        self.gantry_refresh_btn = QPushButton("Refresh\nPorts")
        self.gantry_refresh_btn.clicked.connect(self.update_gantry_ports)
        gantry_layout.addWidget(self.gantry_refresh_btn, 0, 2)
        
        # "Send Positional Command" button
        self.gantry_command_btn = QPushButton("Send Positional Command")
        self.gantry_command_btn.clicked.connect(self.send_positional_command)
        gantry_layout.addWidget(self.gantry_command_btn, 1, 0, 1, 3)

        # Sliders + LCD for X, Y, Z
        labels = ["Gantry X", "Gantry Y", "Gantry Z"]
        gantry_range = [[10,500],[10,600],[10,300]]
        gantry_homing_pos = [230, 260, 140]
        for i, lbl in enumerate(labels):
            row = i + 2
            label = QLabel(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(gantry_range[i][0], gantry_range[i][1])
            slider.setValue(gantry_homing_pos[i])

            lcd = QLCDNumber()
            lcd.display(gantry_homing_pos[i])
            slider.valueChanged.connect(lcd.display)
            slider.valueChanged.connect(lambda val, idx=i: self.update_gantry(idx, val))

            gantry_layout.addWidget(label,  row, 0)
            gantry_layout.addWidget(slider, row, 1)
            gantry_layout.addWidget(lcd,    row, 2)
        
        # Position X btn
        self.position_x_btn = QPushButton("Positional X")
        self.position_x_btn.clicked.connect(self.position_x)
        gantry_layout.addWidget(self.position_x_btn, 6, 0, 1, 3)
        self.position_x_timer = QTimer(self)
        self.position_x_timer.timeout.connect(self.position_x_sender)

        # Homing btn
        self.gantry_home_btn = QPushButton("HOME")
        self.gantry_home_btn.clicked.connect(self.gantry_home)
        gantry_layout.addWidget(self.gantry_home_btn, 5, 0, 1, 3)

        # Inject all btn
        self.inject_all_btn = QPushButton("INJECT ALL")
        self.inject_all_btn.clicked.connect(self.inject_all)
        gantry_layout.addWidget(self.inject_all_btn, 7, 0, 1, 3)

        # liver biopsy btn
        self.liver_biopsy_btn = QPushButton("LIVER BIOPSY")

        self.liver_biopsy_btn.setStyleSheet("""
            QPushButton {
                color: red;
                font-size: 22px;
                font-weight: bold;
                min-height: 80px;
            }
        """)
        self.liver_biopsy_btn.clicked.connect(self.liver_biopsy)
        gantry_layout.addWidget(self.liver_biopsy_btn, 8, 0, 1, 3)

        control_layout.addWidget(self.gantry_group)

        # Connect port selection signal to instantiate/reconnect Gantry
        self.gantry_port_combo.currentIndexChanged.connect(self.on_gantry_port_changed)

        # Populate port list on startup
        self.update_gantry_ports()

        # ------------------ End Effector Control Pannel ------------------

        self.effector_group = QGroupBox("End Effector Control")
        self.effector_group.setFont(font_label)
        eff_layout = QGridLayout(self.effector_group)

        eff_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.effector_port_combo = QComboBox()
        self.effector_port_combo.addItem("Waiting")
        eff_layout.addWidget(self.effector_port_combo, 0, 1)

        self.effector_refresh_btn = QPushButton("Refresh\nPorts")
        self.effector_refresh_btn.clicked.connect(self.update_effector_ports)
        eff_layout.addWidget(self.effector_refresh_btn, 0, 2)
        
        self.effector_command_btn = QPushButton("Send Effector Command")
        self.effector_command_btn.clicked.connect(self.send_effector_command)
        eff_layout.addWidget(self.effector_command_btn, 1, 0, 1, 3)

        # Sliders for Theta, Delta
        eff_labels = ["Theta", "Delta"]
        eff_range = [[0,180],[0,1]]
        for i, lbl in enumerate(eff_labels):
            row = i + 2
            label = QLabel(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(eff_range[i][0], eff_range[i][1])      # example range, adjust as needed
            slider.setValue(0)           # default
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            lcd = QLCDNumber()
            lcd.display(0)

            # Connect signals
            slider.valueChanged.connect(lcd.display)
            slider.valueChanged.connect(lambda val, idx=i: self.update_effector(idx, val))

            eff_layout.addWidget(label,  row, 0)
            eff_layout.addWidget(slider, row, 1)
            eff_layout.addWidget(lcd,    row, 2)

        left_layout.addWidget(self.effector_group)

        # React to port changes
        self.effector_port_combo.currentIndexChanged.connect(self.on_effector_port_changed)
        self.update_effector_ports()  # Populate combo box

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
        camera_controls_layout.addWidget(self.cam_toggle_btn, stretch=2)

        self.cam1_tune_btn = QPushButton("Color Mask 1")
        self.cam1_tune_btn.setMaximumHeight(50)
        self.cam1_tune_btn.setFont(font_button_1)
        self.cam1_tune_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        camera_controls_layout.addWidget(self.cam1_tune_btn, stretch=1)

        self.cam2_tune_btn = QPushButton("Color Mask 2")
        self.cam2_tune_btn.setMaximumHeight(50)
        self.cam2_tune_btn.setFont(font_button_1)
        self.cam2_tune_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        camera_controls_layout.addWidget(self.cam2_tune_btn, stretch=1)

        self.plot_btn = QPushButton("Show Y Position Plot")
        self.plot_btn.setMaximumHeight(50)
        self.plot_btn.setFont(font_button_1)
        self.plot_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
        self.plot_btn.clicked.connect(self.create_plot_window)
        camera_controls_layout.addWidget(self.plot_btn, stretch=1)

        right_layout.addLayout(camera_controls_layout)

        # Connect camera signals
        self.camera1.frame_ready.connect(self.update_camera1_view)
        self.camera2.frame_ready.connect(self.update_camera2_view)

        # Detection status labels
        self.camera1.detection_update.connect(self.update_detection_status)
        self.camera2.detection_update.connect(self.update_detection_status)
        
        # For Camera 1 group
        self.cam1_detection_status = QLabel("No Target")
        self.cam1_detection_status.setStyleSheet("color: red; font-weight: bold;")
        cam1_layout.addWidget(self.cam1_detection_status)

        # For Camera 2 group
        self.cam2_detection_status = QLabel("No Target")
        self.cam2_detection_status.setStyleSheet("color: red; font-weight: bold;")
        cam2_layout.addWidget(self.cam2_detection_status)

        # Connect camera buttons
        self.cam_toggle_btn.toggled.connect(self.toggle_cameras)
        self.cam1_tune_btn.clicked.connect(self.camera1.open_tuner)
        self.cam2_tune_btn.clicked.connect(self.camera2.open_tuner)
        
        # Camera update timer
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frames)
        
        # Detection state
        self.detection_active = False

    ###### CAMERA GUI CONTROLS #####

    def toggle_cameras(self, checked):
        if checked:
            self.camera1.start()
            self.camera2.start()
            self.cam_toggle_btn.setText("Close Cameras")
            self.camera_timer.start(30)  # ~30 FPS
        else:
            self.camera_timer.stop()
            self.camera1.stop()
            self.camera2.stop()
            self.cam_toggle_btn.setText("Open Cameras")
            self.cam_label1.clear()
            self.cam_label2.clear()

    def update_camera_frames(self):
        self.camera1.update_frame()
        self.camera2.update_frame()

    def update_camera1_view(self, image):
        self.cam_label1.setPixmap(QPixmap.fromImage(image).scaled(
            self.cam_label1.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_camera2_view(self, image):
        self.cam_label2.setPixmap(QPixmap.fromImage(image).scaled(
            self.cam_label2.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_detection_status(self, cam_name, detected, center):
        if cam_name == "Camera 1":
            label = self.cam1_detection_status
        else:
            label = self.cam2_detection_status
            
        if detected:
            cx, cy = center
            label.setText(f"Target at ({cx}, {cy})")
            label.setStyleSheet("color: green;")
            if cam_name == "Camera 1":
                self.target_y_history.append(cx)
                # Keep only the most recent points
                if len(self.target_y_history) > self.max_history_length:
                    self.target_y_history.pop(0)
        else:
            label.setText("No Target")
            label.setStyleSheet("color: red;")

    def start_detection(self):
        self.detection_active = not self.detection_active
        self.camera1.toggle_detection(self.detection_active)
        self.camera2.toggle_detection(self.detection_active)
        
        if self.detection_active:
            self.detection_btn.setText("Stop Target Detection")
        else:
            self.detection_btn.setText("Start Target Detection")
    

    ##### GANTRY CONTROL #####
    def on_gantry_port_changed(self):
        """
        Whenever the user picks a different port in the combo, 
        create/update the Gantry object immediately.
        """
        port = self.gantry_port_combo.currentText()
        if port in ["No Ports Found", "Waiting"] or not port:
            return

        if not self.gantry:
            # If we don't have a Gantry yet, create one
            self.gantry = Gantry(port=port)
            self.hw.gantry = self.gantry  # Update the hardware manager with the new Gantry, weird workaround
        else:
            # Otherwise, update the existing Gantry's port
            self.gantry.set_port(port)

    def update_gantry_ports(self):
        """Refresh the combo box with available serial ports."""
        self.gantry_port_combo.blockSignals(True)
        self.gantry_port_combo.clear()
        ports = list_ports.comports()
        if not ports:
            self.gantry_port_combo.addItem("No Ports Found")
        else:
            for p in ports:
                self.gantry_port_combo.addItem(p.device)
        self.gantry_port_combo.blockSignals(False)

        # Force the logic once to set/instantiate Gantry if there's a port
        if self.gantry_port_combo.count() > 0:
            self.on_gantry_port_changed()

    def send_positional_command(self):
        """Pressing this button calls GOTO on the current Gantry object."""
        if not self.gantry:
            return
        self.gantry.goto_position(self.gantry_x, self.gantry_y, self.gantry_z)

    def update_gantry(self, idx, value):
        """Update internal x,y,z from the sliders."""
        if idx == 0:
            self.gantry_x = value
        elif idx == 1:
            self.gantry_y = value
        elif idx == 2:
            self.gantry_z = value

    def position_x(self):
        timeout = 35
        manager = self.hw
        center_y = 240
        start_time = time.time()

        while True:
            manager.camera1.update_frame()
            if manager.camera1.camera.target_found:
                _, target_y = manager.camera1.camera.get_center_of_mask()
                if abs(target_y - center_y) < 15:
                    print("Target aligned — exiting alignment loop.")
                    break

            if time.time() - start_time > timeout:
                print("Alignment timeout — proceeding to next step.")
                break

            manager.blind_x_control()
            time.sleep(0.5)

    def position_x_sender(self):
        # pixel: y 0-400, set cy=200
        cy = 200
        kp_x = 0.1
        if self.camera1.last_center:
            _ ,target_x = self.camera1.last_center
            error = target_x - cy
            if self.x_error is None:
                self.x_error = error
            elif error - self.x_error > 100:
                print("detection error")
            else:
                print(self.x_error)
                self.gantry_x = self.gantry_x + kp_x * self.x_error
                self.gantry_x = np.clip(self.gantry_x, -100, 200)
                # self.send_positional_command()
        if self.x_error is not None and self.x_error < 10:
            self.position_x_timer.stop()
            print("target Reached!")

    def gantry_home(self):
        if self.gantry:
            self.gantry.home()
            gantry_homing_pos = [230, 260, 140]
            self.gantry_x = gantry_homing_pos[0]
            self.gantry_y = gantry_homing_pos[1]
            self.gantry_z = gantry_homing_pos[2]

    def inject_all(self):
        # STEP 4a: Inject gantry
        print("Step 4a: Injecting gantry.")
        self.hw.gantry.injectA()
        time.sleep(2.5)

        # STEP 4b: Inject both
        print("Step 4b: Injecting both.")
        self.hw.inject_all()
        time.sleep(2.5)

        # STEP 4c: Retract
        print("Step 4c: Retracting sample.")
        self.hw.gantry.injectC()
        time.sleep(10)

    def liver_biopsy(self):
        manager = self.hw

        # Home devices
        manager.home_all()
        time.sleep(35)

        # Move to X start and set blind values
        manager.gantry.goto_position(175, 260, 140)
        manager.set_blind_vals(175, 260, 140)

        # Start camera1 and processing
        manager.camera1.start()
        manager.camera1.camera.processing_active = True
        manager.camera1.detection_active = True
        manager.camera1.gui_active = True

        # Alignment loop
        print("Beginning alignment loop...")
        alignment_active = True
        start_time = time.time()
        center_y = manager.camera1.camera.height // 2
        timeout = 35

        while True:
            manager.camera1.update_frame()
            if manager.camera1.camera.target_found:
                _, target_y = manager.camera1.camera.get_center_of_mask()
                if abs(target_y - center_y) < 15:
                    print("Target aligned — exiting alignment loop.")
                    break

            if time.time() - start_time > timeout:
                print("Alignment timeout — proceeding to next step.")
                break

            manager.blind_x_control()
            time.sleep(0.5)

        time.sleep(1)

        # Start Camera2 and add internal&external parameters
        manager.camera2.start()
        manager.camera2.camera.processing_active = True
        manager.camera2.detection_active = True
        manager.camera2.gui_active = True

        dir_path = os.path.dirname(os.path.realpath(__file__))
        camera_internal_file = dir_path + "/camera2_calibration_data.npz"
        manager.camera2.camera.camera_intrinsics = np.load(camera_internal_file)
        camera_external_file = dir_path + "/cam2_external_parameters_0424.npz"
        manager.camera2.camera.camera_extrinsics = np.load(camera_external_file)["T_world_camera"]

        time.sleep(1)

        # === BREATHING CAPTURE PHASE ===
        print("Capturing breathing motion for stability window using Camera 2...")
        breathing_x_values = []
        breathing_z_values = []
        timestamps = []
        breathing_duration = 15  # Two full cycles
        breath_capture_start = time.time()

        while time.time() - breath_capture_start < breathing_duration:
            manager.camera2.update_frame()
            target = manager.camera2.camera.get_center_of_mask()
            if target:
                target_x, target_z = target
                breathing_x_values.append(target_x)
                breathing_z_values.append(target_z)
                timestamps.append(time.time() - breath_capture_start)
                print(f"[{time.time() - breath_capture_start:.2f}s] Target X: {target_x:.2f}")
            time.sleep(0.05)

        # Convert to numpy arrays
        x_array = np.array(breathing_x_values)
        z_array = np.array(breathing_z_values)
        t_array = np.array(timestamps)

        # === Dynamically filter outliers (MAD-based) ===
        median = np.median(x_array)
        mad = np.median(np.abs(x_array - median))
        z_thresh = 5.0
        valid_mask = np.abs(x_array - median) / (mad + 1e-6) < z_thresh
        x_vals = x_array[valid_mask]
        times = t_array[valid_mask]

        # === Smooth X values ===
        from scipy.ndimage import gaussian_filter1d
        x_vals_smooth = gaussian_filter1d(x_vals, sigma=2)

        # === Auto-tune threshold using peak and valley stats ===
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(x_vals_smooth, distance=10, prominence=1)
        valleys, _ = find_peaks(-x_vals_smooth, distance=10, prominence=1)

        stable_start_time = None
        valley_mean = None
        if len(peaks) == 0 or len(valleys) == 0:
            print("Not enough peaks/valleys to determine threshold. Proceeding without timing.")
        else:
            peak_mean = np.mean(x_vals_smooth[peaks])
            valley_mean = np.mean(x_vals_smooth[valleys])
            alpha = 0.6
            x_thresh = valley_mean + alpha * (peak_mean - valley_mean)
            print(f"Auto-tuned X threshold: {x_thresh:.2f} (valley={valley_mean:.2f}, peak={peak_mean:.2f})")

            # Find 5 consecutive frames above threshold
            min_consecutive = 5
            count = 0
            for i, val in enumerate(x_vals_smooth):
                if val > x_thresh:
                    count += 1
                    if count == min_consecutive:
                        # stable_start_time = times[i - min_consecutive + 1]
                        stable_index = i - min_consecutive + 1
                        time_since_capture_end = t_array[-1] - times[stable_index]
                        stable_start_time = time.time() + time_since_capture_end
                        print(f"Stable breathing window detected at {stable_start_time:.2f}s")
                        break
                else:
                    count = 0

        if stable_start_time is None:
            print("No stable breathing window detected via dynamic threshold.")
            print("Captured X values:", x_array.tolist())


        # === STEP 2: Move to Y/Z ===
        print("Step 2: Sending Y/Z position phase 1.")
        alignment_active = False
        use_preset_yz = True
        if use_preset_yz or valley_mean == None:
            gantry_des_y = 90
            # gantry_des_z = 80 # test
            gantry_des_z = 170 # real
        else:
            # # Test
            # u_target_valley = 623
            # v_target_mean = 440
            # dis_cam2target = 300
            # Real
            v_target_mean = np.mean(z_array)
            u_target_valley = valley_mean
            dis_cam2target = manager.current_x + 67 + 170 # mm, 67 is cam2->gantry x0, 170 is needle->gantry x0
            target_world = self.camera2.camera.get_world_3d(u_target_valley, v_target_mean, dis_cam2target/1000) # unit: m
            print("-"*10, target_world,"-"*10)
            target_world_y = target_world[0]*1000 + 273 # mm, 273 is ArUco_tag_0 -> gantry y0
            target_world_z = - target_world[1]*1000 + 50 # mm, 50 is ArUco_tag_0 -> gantry z0
            needle_length = 276.0 + 40.0 # mm, 276 from rotation center to external end, 40 is the injection of internal
            inject_move = 50.0 / 1.732 * 2 # 50 from INJECTA in Gantry_control.ino 
            # ee_2_y0 = 96, ee_2_z0 = 40 (below z0)

            gantry_des_y = target_world_y - (needle_length + inject_move) / 2.0 * 1.732 - 96
            gantry_des_z = target_world_z - (needle_length + inject_move) / 2.0 + 40

        print(f"The desired Y and Z positions for gantry are: {gantry_des_y}, {gantry_des_z}")
        manager.send_yz_position(y=int(gantry_des_y), z=int(gantry_des_z))
        time.sleep(10)

        # === STEP 3: Send theta ===
        print("Step 3: Sending theta to end effector.")
        manager.send_theta_to_effector(theta=120.0, delta=0.0)
        time.sleep(5)

        # === Wait for stable breathing window ===
        print("Waiting for live valley-to-peak breathing window...")
        entered_valley = False
        consecutive = 0
        max_wait = 10  # seconds
        start_wait = time.time()

        while time.time() - start_wait < max_wait:
            manager.camera2.update_frame()
            target = manager.camera2.camera.get_center_of_mask()
            if target:
                target_x, _ = target
                print(f"[{time.time() - start_wait:.2f}s] Live X: {target_x:.2f}")

                if not entered_valley:
                    if target_x < x_thresh:
                        print("Valley detected, now watching for peak rise...")
                        entered_valley = True
                else:
                    if target_x > x_thresh:
                        consecutive += 1
                        if consecutive >= 5:
                            print("Stable rise out of valley detected. Proceeding with injection.")
                            break
                    else:
                        consecutive = 0
            time.sleep(0.05)

        else:
            print("Timeout waiting for stable valley-to-peak transition — injecting anyway.")


        # === STEP 4: Injection Sequence ===
        print("Step 4a: Injecting gantry.")
        manager.gantry.injectA()
        time.sleep(2)

        print("Step 4b: Injecting both.")
        manager.inject_all()
        time.sleep(2.5)

        print("Step 4c: Retracting sample.")
        manager.gantry.injectC()
        time.sleep(10)

        # === STEP 5: Reset rotation ===
        manager.send_theta_to_effector(theta=0.0, delta=0.0)
        print("Sequence complete.")



   ##### End-Effector Port Handling #####
    def update_effector_ports(self):
        self.effector_port_combo.blockSignals(True)
        self.effector_port_combo.clear()
        ports = list_ports.comports()
        if not ports:
            self.effector_port_combo.addItem("No Ports Found")
        else:
            for p in ports:
                self.effector_port_combo.addItem(p.device)
        self.effector_port_combo.blockSignals(False)
        if self.effector_port_combo.count() > 0:
            self.on_effector_port_changed()

    def on_effector_port_changed(self):
        port = self.effector_port_combo.currentText()
        if port in ["No Ports Found", "Waiting"] or not port:
            return
        if not self.end_effector:
            self.end_effector = EndEffector(port=port)
            self.hw.end_effector = self.end_effector # same as gantry
        else:
            self.end_effector.set_port(port)

    def update_effector(self, idx, value):
        """Update internal angles from the sliders (theta, delta)."""
        if   idx == 0: self.theta = value
        elif idx == 1: self.delta = value

    def send_effector_command(self):
        """Send the current theta/delta to the end effector."""
        if self.end_effector:
            self.end_effector.goto_position(self.theta, self.delta)

    def closeEvent(self, event):
        print("Closing hardware connections...")
        self.hw.close_all()
        event.accept()

    # target y history record and show
    def create_plot_window(self):
        """Create and show a window with a live plot of target y positions."""
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Target Y Position History")
        self.plot_window.resize(800, 400)
        
        # Create chart
        self.chart = QChart()
        self.chart.setTitle("Target Y Position Over Time")
        
        # Create series
        self.series = QLineSeries()
        self.series.setName("Y Position")
        
        # Add data to series
        for i, y in enumerate(self.target_y_history):
            self.series.append(i, y)
        
        self.chart.addSeries(self.series)
        
        # Create axes
        self.axisX = QValueAxis()
        self.axisX.setTitleText("Time (samples)")
        self.axisX.setRange(0, self.max_history_length)
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.series.attachAxis(self.axisX)
        
        self.axisY = QValueAxis()
        self.axisY.setTitleText("Y Position (pixels)")
        self.axisY.setRange(0, 400)  # Assuming camera resolution is 400 pixels tall
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisY)
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.chart_view)
        self.plot_window.setLayout(layout)
        
        # Timer to update the plot
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(10)  # Update every 10ms
        
        self.plot_window.show()

    def update_plot(self):
        """Update the plot with new data."""
        self.series.clear()
        for i, y in enumerate(self.target_y_history):
            self.series.append(i, y)
        
        # Auto-scale Y axis if needed
        if self.target_y_history:
            min_y = min(self.target_y_history)
            max_y = max(self.target_y_history)
            padding = 20  # Add some padding
            self.axisY.setRange(min_y - padding, max_y + padding)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Create hardware manager
    hardware_manager = HardwareManager()
    
    # Create and show the interface
    interface = InterfaceLite(hardware_manager=hardware_manager)
    interface.show()

    hardware_manager.close_all()
    sys.exit(app.exec())