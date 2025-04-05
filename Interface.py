import sys
import serial.tools.list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QComboBox, QLCDNumber, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QGuiApplication, QFont
import cv2

class RobotControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
        left_layout.addLayout(control_layout, stretch=1)
        
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
        self.gantry_gui_toggle = QPushButton("Disable GUI Control")
        self.gantry_gui_toggle.setMaximumHeight(50)
        self.gantry_gui_toggle.setCheckable(True)
        self.gantry_gui_toggle.setChecked(True)
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

        control_layout.addWidget(self.gantry_group)
        
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
        self.endeff_gui_toggle = QPushButton("Disable GUI Control")
        self.endeff_gui_toggle.setMaximumHeight(50)
        self.endeff_gui_toggle.setCheckable(True)
        self.endeff_gui_toggle.setChecked(True)
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

        # empty_label = QLabel()
        # endeff_layout.addWidget(empty_label, 4, 0, 1, 3)

        control_layout.addWidget(self.endeff_group)
        
        # ------------------ Left Bottom: Button Groups ------------------
        button_layout = QHBoxLayout()
        left_layout.addLayout(button_layout, stretch=1)
        
        # Motor Homing Group
        motor_home_group = QGroupBox("Motor Homing")
        motor_home_group.setFont(font_label)
        motor_home_group.setStyleSheet("QGroupBox * { font-size: 14px; font-weight: normal; }")
        motor_home_layout = QVBoxLayout()
        motor_home_group.setLayout(motor_home_layout)
        self.motor_home_btn = QPushButton("Motor Homing")
        self.motor_home_btn.setMaximumHeight(50)
        self.motor_home_btn.clicked.connect(self.motor_home)

        motor_home_status_layout = QHBoxLayout()
        fixed_status_label = QLabel("Status:")
        fixed_status_label.setFixedWidth(80)
        self.motor_home_status = QLabel("Not Homed")
        motor_home_status_layout.addWidget(fixed_status_label)
        motor_home_status_layout.addWidget(self.motor_home_status)

        motor_home_layout.addWidget(self.motor_home_btn)
        motor_home_layout.addLayout(motor_home_status_layout)
        button_layout.addWidget(motor_home_group)
        
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

        # Left Camera status: fixed and dynamic labels side by side
        left_status_layout = QHBoxLayout()
        left_fixed_label = QLabel("Left Camera:")
        self.left_detection_status = QLabel("Not Detected")
        left_status_layout.addWidget(left_fixed_label)
        left_status_layout.addWidget(self.left_detection_status)
        detection_layout.addLayout(left_status_layout)

        # Right Camera status: fixed and dynamic labels side by side
        right_status_layout = QHBoxLayout()
        right_fixed_label = QLabel("Right Camera:")
        self.right_detection_status = QLabel("Not Detected")
        right_status_layout.addWidget(right_fixed_label)
        right_status_layout.addWidget(self.right_detection_status)
        detection_layout.addLayout(right_status_layout)

        button_layout.addWidget(detection_group)

        # Liver Boipsy Group (with 4-step status)
        self.liver_boipsy_group = QGroupBox("Liver Boipsy")
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

        button_layout.addWidget(self.liver_boipsy_group)
        
        # ------------------ Timer and Camera Initialization ------------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_views)
        self.cameras_active = False
        
        # Initialize cameras (using OpenCV for demonstration)
        self.cap1 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(1)
        
        # GUI control enable flags
        self.gantry_gui_enabled = True
        self.endeff_gui_enabled = True

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
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            if ret1:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                h, w, ch = frame1.shape
                bytes_per_line = ch * w
                image1 = QImage(frame1.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.cam_label1.setPixmap(QPixmap.fromImage(image1).scaled(self.cam_label1.size(), Qt.AspectRatioMode.KeepAspectRatio))
            if ret2:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                h, w, ch = frame2.shape
                bytes_per_line = ch * w
                image2 = QImage(frame2.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.cam_label2.setPixmap(QPixmap.fromImage(image2).scaled(self.cam_label2.size(), Qt.AspectRatioMode.KeepAspectRatio))
    
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
        else:
            self.gantry_gui_toggle.setText("Enable GUI Control")
    
    def toggle_endeff_gui(self, checked):
        self.endeff_gui_enabled = checked
        if checked:
            self.endeff_gui_toggle.setText("Disable GUI Control")
        else:
            self.endeff_gui_toggle.setText("Enable GUI Control")
    
    def update_gantry_ports(self):
        ports = serial.tools.list_ports.comports()
        self.gantry_port_combo.clear()
        port_names = [port.device for port in ports]
        if not port_names:
            port_names = ["No ports available"]
        self.gantry_port_combo.addItems(port_names)
        print("Gantry ports updated:", port_names)
    
    def update_endeff_ports(self):
        ports = serial.tools.list_ports.comports()
        self.endeff_port_combo.clear()
        port_names = [port.device for port in ports]
        if not port_names:
            port_names = ["No ports available"]
        self.endeff_port_combo.addItems(port_names)
        print("End-Effector ports updated:", port_names)
    
    def motor_home(self):
        print("Executing motor homing")
        self.motor_home_status.setText("Homing...")
        QTimer.singleShot(2000, lambda: self.motor_home_status.setText("Homed"))
    
    def start_detection(self):
        print("Starting target detection process")
        self.left_detection_status.setText("Detecting...")
        self.right_detection_status.setText("Detecting...")
        QTimer.singleShot(3000, lambda: (
            self.left_detection_status.setText("Target Detected"),
            self.left_detection_status.setStyleSheet("color: green;")
        ))
        QTimer.singleShot(3000, lambda: (
            self.right_detection_status.setText("No Target"),
            self.right_detection_status.setStyleSheet("color: red;")
        ))
        
    def start_liver_boipsy(self):
        print("Starting Liver Boipsy process")
        for lbl in self.liver_status_labels:
            lbl.setText("Wait")
        QTimer.singleShot(1000, lambda: (self.liver_status_labels[0].setText("Pass"), self.liver_status_labels[0].setStyleSheet("color: green;")))
        QTimer.singleShot(2000, lambda: (self.liver_status_labels[1].setText("Pass"), self.liver_status_labels[1].setStyleSheet("color: green;")))
        QTimer.singleShot(3000, lambda: (self.liver_status_labels[2].setText("Pass"), self.liver_status_labels[2].setStyleSheet("color: green;")))
        QTimer.singleShot(4000, lambda: (self.liver_status_labels[3].setText("Pass"), self.liver_status_labels[3].setStyleSheet("color: green;")))
    
    def closeEvent(self, event):
        self.cap1.release()
        self.cap2.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotControlWindow()
    window.show()
    sys.exit(app.exec())
