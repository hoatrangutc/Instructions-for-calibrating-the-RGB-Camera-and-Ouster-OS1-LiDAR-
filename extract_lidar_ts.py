# extract_lidar_ts.py - lưu timestamps của lidar
import rosbag
import numpy as np

bag        = rosbag.Bag("dulieu_cam_lidar_2025-05-29-15-39-10.bag")
timestamps = []

for topic, msg, t in bag.read_messages(topics=["/os1_cloud_node/points"]):
    timestamps.append(t.to_nsec())

bag.close()

np.save("lidar_timestamps.npy", np.array(timestamps))
print(f"✓ Saved {len(timestamps)} LiDAR timestamps")