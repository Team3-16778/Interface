# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# A simple code snippet
# Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit with two CSI ports (Jetson Nano, Jetson Xavier NX) via OpenCV
# Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# This script will open a window and place the camera stream from each camera in a window
# arranged horizontally.
# The camera streams are each read in their own thread, as when done sequentially there
# is a noticeable lag

import os
import cv2
import threading
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        # Internal Parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        # External Parameters
        self.Rotation_matrix = None
        self.Translation_vector = None

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

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

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080
"""


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3280,
    capture_height=2464,
    display_width=1920,
    display_height=1080,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def run_cameras():
    window_title = "Dual CSI Cameras"
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=3280,
            capture_height=2464,
            flip_method=0,
            framerate=21,
        )
    )
    int_params_file = "camera2_calibration_data.npz"
    left_camera.get_internal_parameters(int_params_file)
    ext_params_file = "cam2_external_parameters_2.npz"
    left_camera.get_external_parameters(ext_params_file)
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            capture_width=3280,
            capture_height=2464,
            flip_method=0,
            framerate=21,
        )
    )
    right_camera.start()

    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                _, left_image = left_camera.read()
                left_image, left_image_mask, left_image_mask_bgr = colormask(left_image, left_camera)
                _, right_image = right_camera.read()
                # Use numpy to place images next to each other
                camera_images = np.hstack((left_image, left_image_mask_bgr)) 
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, camera_images)
                else:
                    break

                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        finally:

            left_camera.stop()
            left_camera.release()
            right_camera.stop()
            right_camera.release()
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()

def calculate_world_3D(camera, u, v, Zc = 0.5): # Zc is the distance from the camera to the target(default is a random value)
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
    fx = camera.camera_matrix[0, 0]
    fy = camera.camera_matrix[1, 1]
    cx = camera.camera_matrix[0, 2]
    cy = camera.camera_matrix[1, 2]
    Rotation_matrix = camera.Rotation_matrix
    Translation_vector = camera.Translation_vector

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


# # Dark Spot Color Mask
# def colormask(image, camera):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     inv = cv2.bitwise_not(gray)

#     # Blob detector setup
#     params = cv2.SimpleBlobDetector_Params()
#     params.filterByColor = True
#     params.blobColor = 255

#     params.filterByArea = True
#     params.minArea = 20
#     params.maxArea = 1000

#     params.filterByCircularity = True
#     params.minCircularity = 0.5

#     params.filterByConvexity = False
#     params.filterByInertia = False

#     detector = cv2.SimpleBlobDetector_create(params)
#     keypoints = detector.detect(inv)

#     # Sort blobs by size (descending) and pick the largest one
#     keypoints = sorted(keypoints, key=lambda k: -k.size)
#     if keypoints:
#         kp = keypoints[0]
#         x, y = int(kp.pt[0]), int(kp.pt[1])

#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         output_image = cv2.drawKeypoints(image_rgb, [kp], np.array([]),
#                                          (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         mask = np.zeros_like(gray)
#         cv2.circle(mask, (x, y), int(kp.size / 2), 255, -1)
#         mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

#         print(f"Target blob at ({x}, {y})")

#         # Calculate the world coordinate of the Target
#         world_coords = calculate_world_3D(camera, x, y)
#         # Add a label
#         label = "({:.2f}, {:.2f}, {:.2f})".format(world_coords[0], world_coords[1], world_coords[2])
#         cv2.putText(image_rgb, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, (0, 255, 0), 2)
#         print(f"World coords: X={world_coords[0]:.2f}, Y={world_coords[1]:.2f}, Z={world_coords[2]:.2f}")
#     else:
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         output_image = image_rgb.copy()
#         mask_bgr = np.zeros_like(image_rgb)

#     return image_rgb, output_image, mask_bgr


# Red Color Mask
def colormask(image):
    # Preprocessing: blur to reduce noise
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Red hue range (wraps around 0)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Create binary masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    target_find = False
    if contours:
        target_find = True
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Ignore small blobs
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w // 2, y + h // 2

            # Draw bounding box and center
            cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawMarker(mask_bgr, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            print(f"Detected object center: ({cx}, {cy})")

            # # Calculate the world coordinate of the Target
            # world_coords = calculate_world_3D(camera, x, y)
            # # Add a label
            # label = "({:.2f}, {:.2f}, {:.2f})".format(world_coords[0], world_coords[1], world_coords[2])
            # cv2.putText(image_rgb, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.5, (0, 255, 0), 2)
            # print(f"World coords: X={world_coords[0]:.2f}, Y={world_coords[1]:.2f}, Z={world_coords[2]:.2f}")

    return image_rgb, masked_image, mask_bgr, target_find


if __name__ == "__main__":
    run_cameras()
