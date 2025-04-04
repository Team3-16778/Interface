import sys
import serial.tools.list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGridLayout, QComboBox, QLCDNumber
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2

class RobotControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Front-End Control Interface")
        self.resize(1400, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # Main layout: left side for video and control sections, right side for global control buttons
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left layout (cameras, Gantry, and End-Effector sections)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        
        # Right global control layout
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)
        
        # --------------------- Left Section ---------------------
        # Camera section
        cam_group = QGroupBox("Cameras")
        cam_layout = QHBoxLayout()
        cam_group.setLayout(cam_layout)
        self.cam_label1 = QLabel("Camera 1")
        self.cam_label1.setFixedSize(320, 240)
        self.cam_label1.setStyleSheet("background-color: #000;")
        self.cam_label2 = QLabel("Camera 2")
        self.cam_label2.setFixedSize(320, 240)
        self.cam_label2.setStyleSheet("background-color: #000;")
        cam_layout.addWidget(self.cam_label1)
        cam_layout.addWidget(self.cam_label2)
        
        # Camera toggle button
        self.cam_toggle_btn = QPushButton("Open Camera")
        self.cam_toggle_btn.setCheckable(True)
        self.cam_toggle_btn.toggled.connect(self.toggle_cameras)
        
        left_layout.addWidget(cam_group)
        left_layout.addWidget(self.cam_toggle_btn)
        
        # Gantry control section
        gantry_group = QGroupBox("Gantry Control")
        gantry_layout = QGridLayout()
        gantry_group.setLayout(gantry_layout)
        
        # Add USB port selection combobox and refresh button
        gantry_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.gantry_port_combo = QComboBox()
        self.gantry_port_combo.addItems(["No ports available"])
        gantry_layout.addWidget(self.gantry_port_combo, 0, 1)
        self.gantry_refresh_btn = QPushButton("Refresh Port")
        self.gantry_refresh_btn.clicked.connect(self.update_gantry_ports)
        gantry_layout.addWidget(self.gantry_refresh_btn, 0, 2)
        
        # Add GUI control toggle button for Gantry, moved to column 3
        self.gantry_gui_toggle = QPushButton("Enable GUI Control")
        self.gantry_gui_toggle.setCheckable(True)
        self.gantry_gui_toggle.setChecked(True)
        self.gantry_gui_toggle.toggled.connect(self.toggle_gantry_gui)
        gantry_layout.addWidget(self.gantry_gui_toggle, 0, 3)
        
        self.gantry_sliders = []
        self.gantry_lcds = []
        for i in range(3):
            row = i + 1
            label = QLabel(f"Stepper {i+1}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)  # Adjust range as needed
            slider.setValue(50)
            slider.valueChanged.connect(lambda value, idx=i: self.update_gantry(idx, value))
            self.gantry_sliders.append(slider)
            # LCD display to show current slider value
            lcd = QLCDNumber()
            lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
            lcd.display(50)
            # Increase the font size for more visibility
            lcd.setStyleSheet("font-size: 24pt;")
            self.gantry_lcds.append(lcd)
            slider.valueChanged.connect(lcd.display)
            
            gantry_layout.addWidget(label, row, 0)
            gantry_layout.addWidget(slider, row, 1)
            gantry_layout.addWidget(lcd, row, 2, 1, 2)
        
        left_layout.addWidget(gantry_group)
        
        # End-Effector control section
        endeff_group = QGroupBox("End-Effector Control")
        endeff_layout = QGridLayout()
        endeff_group.setLayout(endeff_layout)
        
        # Add USB port selection combobox and refresh button
        endeff_layout.addWidget(QLabel("USB Port:"), 0, 0)
        self.endeff_port_combo = QComboBox()
        self.endeff_port_combo.addItems(["No ports available"])
        endeff_layout.addWidget(self.endeff_port_combo, 0, 1)
        self.endeff_refresh_btn = QPushButton("Refresh Port")
        self.endeff_refresh_btn.clicked.connect(self.update_endeff_ports)
        endeff_layout.addWidget(self.endeff_refresh_btn, 0, 2)
        
        # Add GUI control toggle button for End-Effector, moved to column 3
        self.endeff_gui_toggle = QPushButton("Enable GUI Control")
        self.endeff_gui_toggle.setCheckable(True)
        self.endeff_gui_toggle.setChecked(True)
        self.endeff_gui_toggle.toggled.connect(self.toggle_endeff_gui)
        endeff_layout.addWidget(self.endeff_gui_toggle, 0, 3)
        
        # Servo control slider
        endeff_layout.addWidget(QLabel("Servo Angle"), 1, 0)
        self.servo_slider = QSlider(Qt.Orientation.Horizontal)
        self.servo_slider.setRange(0, 180)  # Example range in degrees
        self.servo_slider.setValue(90)
        self.servo_slider.valueChanged.connect(lambda value: self.update_endeff("servo", value))
        self.servo_lcd = QLCDNumber()
        self.servo_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.servo_lcd.display(90)
        self.servo_lcd.setStyleSheet("font-size: 24pt;")
        self.servo_slider.valueChanged.connect(self.servo_lcd.display)
        endeff_layout.addWidget(self.servo_slider, 1, 1)
        endeff_layout.addWidget(self.servo_lcd, 1, 2, 1, 2)
        
        # Linear actuator control slider
        endeff_layout.addWidget(QLabel("Syringe Position"), 2, 0)
        self.linear_slider = QSlider(Qt.Orientation.Horizontal)
        self.linear_slider.setRange(0, 100)  # Adjust range as needed
        self.linear_slider.setValue(50)
        self.linear_slider.valueChanged.connect(lambda value: self.update_endeff("linear", value))
        self.linear_lcd = QLCDNumber()
        self.linear_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.linear_lcd.display(50)
        self.linear_lcd.setStyleSheet("font-size: 24pt;")
        self.linear_slider.valueChanged.connect(self.linear_lcd.display)
        endeff_layout.addWidget(self.linear_slider, 2, 1)
        endeff_layout.addWidget(self.linear_lcd, 2, 2, 1, 2)
        
        left_layout.addWidget(endeff_group)
        
        # --------------------- Right Global Control Section ---------------------
        # Motor Homing section
        motor_home_group = QGroupBox("Motor Homing")
        motor_home_layout = QVBoxLayout()
        motor_home_group.setLayout(motor_home_layout)
        self.motor_home_btn = QPushButton("Motor Homing")
        self.motor_home_btn.clicked.connect(self.motor_home)
        self.motor_home_status = QLabel("Status: Not Homed")
        motor_home_layout.addWidget(self.motor_home_btn)
        motor_home_layout.addWidget(self.motor_home_status)
        
        right_layout.addWidget(motor_home_group)
        
        # Target Detection section
        detection_group = QGroupBox("Target Detection")
        detection_layout = QVBoxLayout()
        detection_group.setLayout(detection_layout)
        self.detection_btn = QPushButton("Start Target Detection")
        self.detection_btn.clicked.connect(self.start_detection)
        # Labels to show detection status for left and right cameras
        self.left_detection_status = QLabel("Left Camera: Not Detected")
        self.right_detection_status = QLabel("Right Camera: Not Detected")
        detection_layout.addWidget(self.detection_btn)
        detection_layout.addWidget(self.left_detection_status)
        detection_layout.addWidget(self.right_detection_status)
        
        right_layout.addWidget(detection_group)
        
        # New "Biopsy Process" button added to the right section
        self.biopsy_btn = QPushButton("Biopsy Process")
        self.biopsy_btn.clicked.connect(self.start_biopsy)
        right_layout.addWidget(self.biopsy_btn)
        
        right_layout.addStretch()
        
        # --------------------- Camera Timer and Initialization ---------------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_views)
        self.cameras_active = False
        
        # Initialize cameras (using OpenCV for demonstration)
        self.cap1 = cv2.VideoCapture(0)  # Camera 1; device index may need adjustment
        self.cap2 = cv2.VideoCapture(1)  # Camera 2
        
        # Control variables: determine if GUI control is enabled (default is enabled)
        self.gantry_gui_enabled = True
        self.endeff_gui_enabled = True
        
    def toggle_cameras(self, checked):
        if checked:
            self.cam_toggle_btn.setText("Close Camera")
            self.cameras_active = True
            self.timer.start(30)  # Update roughly every 30 milliseconds
        else:
            self.cam_toggle_btn.setText("Open Camera")
            self.cameras_active = False
            self.timer.stop()
            # Clear camera display areas
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
            # Add code here to control the corresponding stepper motor for the Gantry via serial or other interface
            print(f"Gantry Stepper {idx+1} set to {value}")
        else:
            print(f"Gantry GUI control disabled; ignoring change for Stepper {idx+1}")
        
    def update_endeff(self, component, value):
        if self.endeff_gui_enabled:
            # Add code here to control the End-Effector, choosing between servo or linear actuator
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
        # Update Gantry USB port combobox with current available ports
        ports = serial.tools.list_ports.comports()
        self.gantry_port_combo.clear()
        port_names = [port.device for port in ports]
        if not port_names:
            port_names = ["No ports available"]
        self.gantry_port_combo.addItems(port_names)
        print("Gantry ports updated:", port_names)
        
    def update_endeff_ports(self):
        # Update End-Effector USB port combobox with current available ports
        ports = serial.tools.list_ports.comports()
        self.endeff_port_combo.clear()
        port_names = [port.device for port in ports]
        if not port_names:
            port_names = ["No ports available"]
        self.endeff_port_combo.addItems(port_names)
        print("End-Effector ports updated:", port_names)
            
    def motor_home(self):
        # Add code here to send homing commands to all motors
        print("Executing motor homing")
        self.motor_home_status.setText("Status: Homing...")
        # Simulate delay then update status to homed
        QTimer.singleShot(2000, lambda: self.motor_home_status.setText("Status: Homed"))
        
    def start_detection(self):
        # Add code here to trigger the target detection program
        print("Starting target detection process")
        self.left_detection_status.setText("Left Camera: Detecting...")
        self.right_detection_status.setText("Right Camera: Detecting...")
        # Simulate detection process and update statuses
        QTimer.singleShot(3000, lambda: self.left_detection_status.setText("Left Camera: Target Detected"))
        QTimer.singleShot(3000, lambda: self.right_detection_status.setText("Right Camera: No Target"))
        
    def start_biopsy(self):
        # Add code here to start the biopsy process
        print("Starting biopsy process")
        self.biopsy_btn.setText("Biopsy In Process")
        QTimer.singleShot(3000, lambda: self.biopsy_btn.setText("Biopsy Process"))
        
    def closeEvent(self, event):
        # Release camera resources on close
        self.cap1.release()
        self.cap2.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotControlWindow()
    window.show()
    sys.exit(app.exec())
