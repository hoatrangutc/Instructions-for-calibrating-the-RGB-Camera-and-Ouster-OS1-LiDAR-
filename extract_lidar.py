# extract_lidar.py
import rosbag
import numpy as np
from sensor_msgs import point_cloud2
import os

os.makedirs("lidar_frames", exist_ok=True)

bag   = rosbag.Bag("dulieu_cam_lidar_2025-05-29-15-39-10.bag")
count = 0

for topic, msg, t in bag.read_messages(topics=["/os1_cloud_node/points"]):
    try:
        pts = np.array(list(point_cloud2.read_points(
            msg, field_names=("x","y","z"), skip_nans=True
        )), dtype=np.float32)
        
        np.save(f"lidar_frames/{count:06d}.npy", pts)
        del pts
        count += 1
        
        if count % 100 == 0:
            print(f"  {count}/2058 frames...")
            
    except Exception as e:
        print(f"  Frame {count} error: {e}")
        count += 1

bag.close()
print(f"✓ Done: {count} lidar frames")