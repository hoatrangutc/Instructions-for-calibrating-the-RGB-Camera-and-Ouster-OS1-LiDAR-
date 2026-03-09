# extract_frames.py - lưu ảnh ra file trước
import rosbag
from cv_bridge import CvBridge
import cv2, os

os.makedirs("calib_frames", exist_ok=True)
bridge = CvBridge()
bag = rosbag.Bag("dulieu_cam_lidar_2025-05-29-15-39-10.bag")

count = 0
for topic, msg, t in bag.read_messages(topics=["/usb_cam/image_raw"]):
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.imwrite(f"calib_frames/{count:06d}.png", img)
    count += 1

bag.close()
print(f"Extracted {count} frames → {img.shape}")  # in ra resolution