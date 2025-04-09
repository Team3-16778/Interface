import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout,
    QSlider, QSpinBox, QPushButton, QHBoxLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QMouseEvent

class ColorMask(QWidget):
    def __init__(self, camera_name="Camera", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Color Mask Tuner - {camera_name}")
        self.setGeometry(100, 100, 1000, 600)

        self.label = QLabel()
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.pick_color
        self.display_img = None
        self.current_frame = None

        self.sliders = {}
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        grid = QGridLayout()
        hsv_names = ["Low H", "Low S", "Low V", "High H", "High S", "High V"]
        default_values1 = [0, 100, 100, 10, 255, 255]
        default_values2 = [160, 100, 100, 179, 255, 255]

        for col, (suffix, defaults) in enumerate(zip(['1', '2'], [default_values1, default_values2])):
            for row, (name, val) in enumerate(zip(hsv_names, defaults)):
                full_name = f"{name}{suffix}"
                lbl = QLabel(full_name)
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setRange(0, 255 if 'S' in name or 'V' in name else 179)
                sld.setValue(val)
                spn = QSpinBox()
                spn.setRange(0, 255 if 'S' in name or 'V' in name else 179)
                spn.setValue(val)
                sld.valueChanged.connect(spn.setValue)
                spn.valueChanged.connect(sld.setValue)
                sld.valueChanged.connect(self.update_frame)

                self.sliders[full_name] = spn
                grid.addWidget(lbl, row, col * 3 + 0)
                grid.addWidget(sld, row, col * 3 + 1)
                grid.addWidget(spn, row, col * 3 + 2)

        layout.addLayout(grid)
        self.setLayout(layout)

    def get_hsv_bounds(self):
        l1 = np.array([self.sliders[f"Low H1"].value(), self.sliders[f"Low S1"].value(), self.sliders[f"Low V1"].value()])
        u1 = np.array([self.sliders[f"High H1"].value(), self.sliders[f"High S1"].value(), self.sliders[f"High V1"].value()])
        l2 = np.array([self.sliders[f"Low H2"].value(), self.sliders[f"Low S2"].value(), self.sliders[f"Low V2"].value()])
        u2 = np.array([self.sliders[f"High H2"].value(), self.sliders[f"High S2"].value(), self.sliders[f"High V2"].value()])
        return l1, u1, l2, u2

    def set_frame(self, frame):
        self.current_frame = frame.copy()

    def process_frame(self, frame):
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        l1, u1, l2, u2 = self.get_hsv_bounds()

        mask1 = cv2.inRange(hsv, l1, u1)
        mask2 = cv2.inRange(hsv, l2, u2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = frame.copy()

        target_found = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 20)
            cv2.circle(overlay, (cx, cy), 5, (255, 255, 0), 3)
            target_found = True

        return mask, overlay, target_found

    def update_frame(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        mask, overlay, _ = self.process_frame(frame)

        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack((frame, overlay, mask_display))
        display = cv2.resize(display, (960, 320))
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        self.display_img = display_rgb
        h, w, ch = display_rgb.shape
        qimg = QImage(display_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def pick_color(self, event: QMouseEvent):
        if self.display_img is None or self.current_frame is None:
            return

        x_disp = int(event.position().x())
        y_disp = int(event.position().y())

        h_orig, w_orig = self.current_frame.shape[:2]
        panel_width = self.display_img.shape[1] // 3
        if x_disp > panel_width:
            return

        x_ratio = w_orig / panel_width
        y_ratio = h_orig / self.display_img.shape[0]
        x = int(x_disp * x_ratio)
        y = int(y_disp * y_ratio)

        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)

        radius = 2
        x1, x2 = max(0, x - radius), min(w_orig, x + radius + 1)
        y1, y2 = max(0, y - radius), min(h_orig, y + radius + 1)
        patch = hsv[y1:y2, x1:x2]
        median = np.median(patch.reshape(-1, 3), axis=0)

        margin = 20
        low = np.clip(median - margin, [0, 0, 0], [179, 255, 255]).astype(int)
        high = np.clip(median + margin, [0, 0, 0], [179, 255, 255]).astype(int)

        if low[0] <= high[0]:
            self.sliders["Low H1"].setValue(int(low[0]))
            self.sliders["Low S1"].setValue(int(low[1]))
            self.sliders["Low V1"].setValue(int(low[2]))
            self.sliders["High H1"].setValue(int(high[0]))
            self.sliders["High S1"].setValue(int(high[1]))
            self.sliders["High V1"].setValue(int(high[2]))
            self.sliders["Low H2"].setValue(0)
            self.sliders["High H2"].setValue(0)
        else:
            self.sliders["Low H1"].setValue(0)
            self.sliders["High H1"].setValue(int(high[0]))
            self.sliders["Low H2"].setValue(int(low[0]))
            self.sliders["High H2"].setValue(179)

        self.sliders["Low S1"].setValue(int(low[1]))
        self.sliders["Low V1"].setValue(int(low[2]))
        self.sliders["High S1"].setValue(int(high[1]))
        self.sliders["High V1"].setValue(int(high[2]))
        self.sliders["Low S2"].setValue(int(low[1]))
        self.sliders["Low V2"].setValue(int(low[2]))
        self.sliders["High S2"].setValue(int(high[1]))
        self.sliders["High V2"].setValue(int(high[2]))

        self.update_frame()

    def apply(self, frame):
        mask, overlay, found = self.process_frame(frame)
        return mask, overlay, found

    def open_tuner(self):
        self.show()
        self.raise_()
        self.activateWindow()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = ColorMask("Demo Camera")
    cap = cv2.VideoCapture(0)

    def feed():
        ret, frame = cap.read()
        if ret:
            demo.set_frame(frame)

    timer = QTimer()
    timer.timeout.connect(feed)
    timer.start(30)

    demo.show()
    sys.exit(app.exec())
