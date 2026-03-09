# test_undistort.py
import cv2
import numpy as np

# Kết quả calibration
K = np.array([
    [448.9443277836273,  0.0,              328.1278074020808],
    [0.0,               449.8824080758361, 229.13027052795243],
    [0.0,               0.0,              1.0]
])

D = np.array([0.08804103349773844, -0.012905824328900526,
              -0.002744215700176299, -0.007056489723647028,
              -0.4767527602193595])

# Test với 1 ảnh mẫu - đổi path tùy ý
img = cv2.imread("calib_frames/000300.png")

# Undistort
undistorted = cv2.undistort(img, K, D)

# Vẽ grid để thấy rõ distortion correction
def draw_grid(image, step=60, color=(0, 255, 0)):
    h, w = image.shape[:2]
    for x in range(0, w, step):
        cv2.line(image, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(image, (0, y), (w, y), color, 1)
    return image

img_grid  = draw_grid(img.copy())
und_grid  = draw_grid(undistorted.copy())

combined = np.hstack([img_grid, und_grid])

cv2.putText(combined, "Original (distorted)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(combined, "Undistorted  RMS=0.275px EXCELLENT",
            (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("Calibration Check", combined)
cv2.imwrite("calibration_check.png", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu để dùng cho các bước sau
np.save("camera_K.npy", K)
np.save("camera_D.npy", D)
print("✓ Saved camera_K.npy và camera_D.npy")
print(f"\nfx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
print(f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")