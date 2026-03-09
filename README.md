************
HƯỚNG DẪN THU THẬP DỮ LIỆU CAMERA-LIDAR VÀO FILE .BAG 
#Bước 1:
source devel/setup.bash
roslaunch ouster_ros os1.launch os1_udp_dest:=10.5.5.1 os1_hostname:=10.5.5.54
#Bước 2: mở terminal mới mở rviz để quan sát dữ liệu camera và lidar
rviz
#Bước 3: mở terminal để bật camera 
roslaunch usb_cam usb_cam.launch
Bước 4: Vào rviz thêm pointcloud2 và thêm image raw, chọn frame là //os1_lidar

Bước 5:mở terminal mới để ghi dữ liệu
rosbag record /usb_cam/image_raw /os1_cloud_node/points /os1_cloud_node/imu -o dulieu_cam_lidar.bag
ctrl+c để stop
mở lại bằng 
rosbag play dulieu_cam_lidar.bag //thay bằng tên file thực tế

*****************************************************************************************************************************
HƯỚNG DẪN HIỆU CHỈNH CAMERA VÀ LIDAR ĐỂ THU ĐƯỢC MA TRẬN (K, D, R, T)

# Bước 1: Intrinsic (Hiệu chỉnh nội tại cho camera)

Cách 1: Hiệu chỉnh thời gian thực (camera trực tiếp) -- nên dùng

roscore
roslaunch usb_cam usb_cam.launch
chay export DISPLAY=:0 (neu dung VNC)
rosrun camera_calibration cameracalibrator.py --size 6x8 --square 0.059 --camera_name usb_cam image:=/usb_cam/image_raw camera:=/usb_cam

✅ Cách cầm bảng hiệu chỉnh
 • Luôn cầm bảng vuông góc với camera, không cần nghiêng.
 • Không quan trọng bảng ngang hay dọc, quan trọng là:
 • Phải thay đổi vị trí, xoay góc, dịch chuyển xa gần, nghiêng trái/phải, để camera thấy các phối cảnh khác nhau của bảng.
 • Ít nhất 15–20 ảnh với các tư thế khác nhau của bảng.
 
 ✅ Khi nào bấm “CALIBRATE”
 • Khi bạn thấy thanh trạng thái ở dưới (thanh màu vàng) báo hiệu là đã thu được đủ số ảnh (ví dụ: Captured: 22 good).
 • Lúc đó, nút CALIBRATE sẽ sáng lên và bạn bấm vào.
 • Nếu chưa đủ ảnh tốt, nút CALIBRATE sẽ bị mờ.
 
 ✅ Sau khi bấm “CALIBRATE”
 • Đợi khoảng vài giây tới vài chục giây để nó tính toán.
 • Sau khi hoàn thành, bấm SAVE để lưu file hiệu chỉnh.
 • Bấm COMMIT để ghi dữ liệu này về ROS (xuất ra file .yaml).
 
Cách 2:  Hiệu chỉnh Camera từ các frame ảnh xuất ra từ file .bag:
python3 extract_frames.py --> tạo ra thư mục calib_frames chứa các frams ảnh xuất ra từ file .bag
python3 intrinsic_calibrate.py --> tạo ra K và D lưu trong camera_intrinsic.yaml 

# Bước 2: Hiệu chỉnh ngoại với LiDAR
python3 extract_lidar.py --> tạo ra thư mục lidar_frames chứa các file .npy pointclound 
python3 extract_lidar_ts --> tạo ra file lidar_timestamps.npy
python3 extrinsic_calibration.py --> tạo ra R_lidar2cam.npy và T_lidar2cam.npy

# Bước 3: Lưu kết quả cuối cùng gồm K, D, R, T để dùng cho Fusion
python3 save_calib.py
