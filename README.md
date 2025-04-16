# Interface of Discount daVinci
Code of the interface for the Discount daVinci

## GUI
The user interface to control the Discount daVinci.

### Python version (tested)
- Python 3.11

### Dependencies
- PySide6
- pyserial
- numpy
- opencv-python

### Hardware
- Arduino Uno
- CSI Cameras
- Actuators

---

## `utils.py` – System Control Classes

This module defines all the primary hardware interfaces and logic used for vision, motion, and automation control in the Discount daVinci system.

### `Camera`
Represents a single camera (CSI or USB). Handles image capture, HSV-based color masking, and 3D world coordinate projection.

**Notable methods:**
- `start_cap()`, `release()` – Start/stop camera stream.
- `detect_target()` – Run masking to locate object.
- `get_world_3d(u, v)` – Projects 2D to 3D using intrinsics/extrinsics.

**Purpose:** Real-time detection and coordinate tracking for target alignment and interaction.

---

### `CameraHandler`
Qt-compatible wrapper for `Camera` with thread-safe updates, detection signals, and tuning controls.

**Notable methods:**
- `update_frame()` – Handles image acquisition and detection.
- `frame_ready`, `detection_update` – Signals for GUI updates.
- `open_tuner()` – Launches HSV color tuning.

**Purpose:** Allows seamless integration between OpenCV and the GUI's event loop.

---

### `AbstractSerialDevice`
Abstract base class for all serial-controlled components.

**Notable methods:**
- `_open_serial()`, `set_port()` – Manage serial interface.
- `home()`, `inject()` – Common commands across devices.
- `goto_position()` – Abstract method for movement.
- `stop()` – Abstract emergency stop.

**Purpose:** Standardizes behavior for all actuated components (e.g., gantry, end-effector).

---

### `Gantry`
Controls a 3-axis motion stage via serial. Inherits from `AbstractSerialDevice`.

**Notable methods:**
- `goto_position(x, y, z)` – Commands 3D motion.
- `set_target()` / `send_to_target()` – Stages and sends target positions.
- `injectA()`, `injectC()` – Specialized actuation commands.

**Purpose:** Controls precise XY(Z) movement and injection for sample placement or manipulation.

---

### `EndEffector`
Rotational control unit for tooling (e.g., syringes, pickers).

**Notable methods:**
- `goto_position(theta, delta)` – Rotate to specified angles.
- `get_angles()` – Read angular state (if supported).

**Purpose:** Enables orientation and rotation-specific actuation separate from gantry movement.

---

### `HardwareManager`
Central orchestrator for camera, gantry, and effector management. Also provides automation routines.

**Notable methods:**
- `connect_*()` – Safe connection methods.
- `blind_x_control()` / `x_axis_controller()` – Visual servoing control of X-axis.
- `send_yz_position()`, `send_theta_to_effector()` – Targeted commands.
- `inject_all()`, `home_all()` – Multi-device operations.
- `close_all()` – Graceful system shutdown and resource cleanup.

**Purpose:** Manages synchronization and integration between components. Primary interface for automated sequences and system control.

---

## Debug

Run with verbose logging and timers to test connectivity, camera feed, and alignment control.

