#!/usr/bin/env python

import rospy
import cv2
import yaml
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from datetime import datetime

class CameraCalibrator:
    def __init__(self):
        rospy.init_node('camera_calibration_node', anonymous=True)
        self.bridge = CvBridge()

        # Cấu hình checkerboard và kích thước ô vuông (59mm)
        self.checkerboard_size = (6, 8)
        self.square_size = 0.059  # mét

        # Khởi tạo subscriber ảnh
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)

        # Cờ hiệu chỉnh
        self.calibration_done = False

        # Dữ liệu cho hiệu chỉnh
        self.objpoints = []
        self.imgpoints = []
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.checkerboard_size[0]*self.checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Tạo thư mục lưu kết quả
        self.yaml_path = os.path.expanduser("~/home/uavai/trangnguyen1/hieu_chuan_cam/camera_info/calibration.yaml")
        self.video_dir = os.path.expanduser("~/thome/uavai/trangnguyen1/hieu_chuan_cam/camera_info/calibration_videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # Khởi tạo ghi video
        self.video_writer = None
        self.frame_size = None
        self.fps = 20
        self.video_path = None

        rospy.loginfo("Camera calibration node started.")
        rospy.spin()

    def image_callback(self, msg):
        if self.calibration_done:
            return

        # Chuyển đổi ảnh ROS sang OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Ghi video nếu chưa khởi tạo
        if self.video_writer is None:
            h, w = gray.shape
            self.frame_size = (w, h)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = os.path.join(self.video_dir, f"calib_{timestamp}.avi")
            self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.frame_size)
            rospy.loginfo(f"Recording calibration video to: {self.video_path}")

        self.video_writer.write(cv_image)  # Ghi mỗi frame

        # Phát hiện checkerboard
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        if ret:
            rospy.loginfo_throttle(3, "Checkerboard detected")
            self.objpoints.append(self.objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            self.imgpoints.append(corners2)
            cv2.drawChessboardCorners(cv_image, self.checkerboard_size, corners2, ret)

        cv2.imshow("Calibration", cv_image)
        key = cv2.waitKey(1) & 0xFF
