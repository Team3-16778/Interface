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
from utils import Gantry, Camera, CameraHandler



class InterfaceLite(QMainWindow):
    def __init__(self):
        super().__init__()


        ## ------------------ Inner Parameters ------------------

        self.gantry = None
        self.gantry_x = 50
        self.gantry_y = 50
        self.gantry_z = 50

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
        for i, lbl in enumerate(labels):
            row = i + 2
            label = QLabel(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)

            lcd = QLCDNumber()
            lcd.display(50)
            slider.valueChanged.connect(lcd.display)
            slider.valueChanged.connect(lambda val, idx=i: self.update_gantry(idx, val))

            gantry_layout.addWidget(label,  row, 0)
            gantry_layout.addWidget(slider, row, 1)
            gantry_layout.addWidget(lcd,    row, 2)

        control_layout.addWidget(self.gantry_group)

        # Connect port selection signal to instantiate/reconnect Gantry
        self.gantry_port_combo.currentIndexChanged.connect(self.on_gantry_port_changed)

        # Populate port list on startup
        self.update_gantry_ports()

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

        right_layout.addLayout(camera_controls_layout)

        # Camera setup
        self.camera1 = CameraHandler(0, "Camera 1", use_csi=False, sensor_id=0)
        self.camera2 = CameraHandler(1, "Camera 2", use_csi=False, sensor_id=1)
        
        # Connect camera signals
        self.camera1.frame_ready.connect(self.update_camera1_view)
        self.camera2.frame_ready.connect(self.update_camera2_view)
        self.camera1.detection_update.connect(self.update_detection_status)
        self.camera2.detection_update.connect(self.update_detection_status)
        
        # Connect camera buttons
        self.cam_toggle_btn.toggled.connect(self.toggle_cameras)
        self.cam1_tune_btn.clicked.connect(self.camera1.open_tuner)
        self.cam2_tune_btn.clicked.connect(self.camera2.open_tuner)
        
        # Camera update timer
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frames)
        
        # Detection state
        self.detection_active = False





        # ------------------- INIT CLEANUP ------------------
        self.update_gantry_ports()




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

    def closeEvent(self, event):
        """
        Clean up Gantry connection on window close, if desired:
        self.gantry.ser.close() ensures the port is freed.
        """
        if self.gantry and self.gantry.ser and self.gantry.ser.is_open:
            self.gantry.ser.close()
        event.accept()


    ##### WINDOW CLOSE HANDLER #####
    def closeEvent(self, event):
        """Clean up Gantry serial connection when closing the window."""
        if self.gantry:
            self.gantry.disconnect()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InterfaceLite()
    window.show()
    sys.exit(app.exec())
