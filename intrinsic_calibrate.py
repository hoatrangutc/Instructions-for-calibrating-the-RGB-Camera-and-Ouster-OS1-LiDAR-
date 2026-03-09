# calibrate_from_images.py
import cv2
import numpy as np
import glob

PATTERN = (6, 8)
SQUARE  = 0.059

objp = np.zeros((PATTERN[0]*PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE

# ==================== BƯỚC 1: LỌC TRÙNG ====================
print("Bước 1: Lọc ảnh trùng lặp...")

images = sorted(glob.glob("calib_frames/*.png"))
images = images[130:4940]

selected = []
prev_hash = None

for i, path in enumerate(images):
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
        del img, gray
        continue

    small = cv2.resize(gray, (16, 16))
    curr_hash = small.flatten().astype(np.float32)

    if prev_hash is not None:
        if np.mean(np.abs(curr_hash - prev_hash)) < 3.0:
            del img, gray, small, curr_hash
            continue

    selected.append(path)
    prev_hash = curr_hash
    del img, gray, small

    if i % 200 == 0:
        print(f"  Scanning {i}/{len(images)}... selected {len(selected)}")

print(f"✓ Unique frames: {len(selected)}")

# ==================== BƯỚC 2: DETECT CORNERS ====================
print("\nBước 2: Detect corners...")

candidates = []
img_shape  = None

for i, path in enumerate(selected):
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        pts = corners2.reshape(-1, 2)

        center_x = pts[:, 0].mean() / img_shape[0]
        center_y = pts[:, 1].mean() / img_shape[1]
        bbox_w   = pts[:, 0].max() - pts[:, 0].min()
        bbox_h   = pts[:, 1].max() - pts[:, 1].min()
        scale    = (bbox_w * bbox_h) / (img_shape[0] * img_shape[1])
        angle    = np.degrees(np.arctan2(
            pts[PATTERN[0]-1][1] - pts[0][1],
            pts[PATTERN[0]-1][0] - pts[0][0]
        ))

        candidates.append({
            "path":     path,
            "corners":  corners2,
            "center_x": center_x,
            "center_y": center_y,
            "scale":    scale,
            "angle":    angle,
        })

    del img, gray
    if i % 100 == 0:
        print(f"  [{i}/{len(selected)}] found {len(candidates)} candidates")

print(f"✓ Tổng candidates có checkerboard: {len(candidates)}")

# ==================== BƯỚC 3: LỌC CHẤT LƯỢNG ====================
print("\nBước 3: Lọc chất lượng...")

candidates_filtered = [c for c in candidates
    if c["scale"] > 0.015
    and abs(c["angle"]) < 50
]
print(f"✓ Sau lọc chất lượng: {len(candidates_filtered)} candidates")

# ==================== BƯỚC 4: CHỌN 500 FRAMES ĐA DẠNG ====================
print("\nBước 4: Chọn 500 frames đa dạng nhất...")

def pick_diverse_frames(candidates, n=500):
    if len(candidates) <= n:
        print(f"  Chỉ có {len(candidates)} candidates, dùng tất cả")
        return candidates

    features = np.array([
        [
            c["center_x"],
            c["center_y"],
            c["scale"] * 3,
            c["angle"] / 45.0,
            min(c["center_x"], 1 - c["center_x"]) * 2,
            min(c["center_y"], 1 - c["center_y"]) * 2,
        ]
        for c in candidates
    ], dtype=np.float32)

    # Bắt đầu từ 4 góc + tâm ảnh
    seed_targets = np.array([
        [0.1, 0.1, 0, 0, 0, 0],  # góc trên trái
        [0.9, 0.1, 0, 0, 0, 0],  # góc trên phải
        [0.1, 0.9, 0, 0, 0, 0],  # góc dưới trái
        [0.9, 0.9, 0, 0, 0, 0],  # góc dưới phải
        [0.5, 0.5, 0, 0, 0, 0],  # tâm ảnh
    ], dtype=np.float32)

    selected_idx = set()
    for target in seed_targets:
        dists = np.linalg.norm(features - target, axis=1)
        idx = int(np.argmin(dists))
        selected_idx.add(idx)

    selected_list = list(selected_idx)

    # Farthest point sampling
    # Tính distance matrix từng bước (tiết kiệm RAM)
    min_dists = np.full(len(candidates), np.inf)

    for idx in selected_list:
        d = np.linalg.norm(features - features[idx], axis=1)
        min_dists = np.minimum(min_dists, d)

    for step in range(n - len(selected_list)):
        # Mask đã chọn
        min_dists[list(selected_idx)] = -1

        next_idx = int(np.argmax(min_dists))
        selected_idx.add(next_idx)
        selected_list.append(next_idx)

        # Update min distances
        d = np.linalg.norm(features - features[next_idx], axis=1)
        min_dists = np.minimum(min_dists, d)

        if step % 100 == 0:
            print(f"  Sampling {step+len(seed_targets)}/{n}...")

    return [candidates[i] for i in sorted(selected_idx)]

diverse = pick_diverse_frames(candidates_filtered, n=500)

angles   = [c["angle"]    for c in diverse]
center_x = [c["center_x"] for c in diverse]
center_y = [c["center_y"] for c in diverse]
scales   = [c["scale"]    for c in diverse]

print(f"\nThống kê {len(diverse)} frames được chọn:")
print(f"  Góc nghiêng : {min(angles):.1f}° → {max(angles):.1f}°")
print(f"  Vị trí tâm X: {min(center_x):.2f} → {max(center_x):.2f}")
print(f"  Vị trí tâm Y: {min(center_y):.2f} → {max(center_y):.2f}")
print(f"  Scale board  : {min(scales):.3f} → {max(scales):.3f}")

# ==================== BƯỚC 5: CALIBRATE ====================
print("\nBước 5: Calibrate...")

objpoints = [objp] * len(diverse)
imgpoints = [c["corners"] for c in diverse]

ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print(f"\nRMS error: {ret:.4f} px  ({'✓ Tốt' if ret < 1.0 else '⚠️ Cao'})")
print(f"\nK =\n{K}")
print(f"D = {D.flatten()}")

# ==================== BƯỚC 6: UNDISTORT CHECK ====================
print("\nBước 6: Kiểm tra undistort (nhấn phím bất kỳ để xem tiếp, ESC thoát)...")

# Chọn 5 frames đại diện: 4 góc + tâm
check_indices = [0, len(diverse)//4, len(diverse)//2, 3*len(diverse)//4, -1]

for idx in check_indices:
    c   = diverse[idx]
    img = cv2.imread(c["path"])
    undistorted = cv2.undistort(img, K, D)

    combined = np.hstack([img, undistorted])
    combined = cv2.resize(combined, (1280, 360))

    fname = c["path"].split("/")[-1]
    cv2.putText(combined, f"Original: {fname}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, f"Undistorted  RMS={ret:.3f}px",
                (660, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Vẽ grid lines để dễ thấy distortion
    h, w = combined.shape[:2]
    for x in range(0, w, 80):
        cv2.line(combined, (x, 0), (x, h), (0, 50, 0), 1)
    for y in range(0, h, 60):
        cv2.line(combined, (0, y), (w, y), (0, 50, 0), 1)

    cv2.imshow("Check undistort", combined)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()

# ==================== LƯU KẾT QUẢ ====================
np.save("camera_K.npy", K)
np.save("camera_D.npy", D)

print("\n" + "="*50)
print("✓ Saved camera_K.npy và camera_D.npy")
print("="*50)
print(f"\nTóm tắt kết quả:")
print(f"  Resolution : {img_shape[0]} x {img_shape[1]}")
print(f"  RMS error  : {ret:.4f} px")
print(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
print(f"  cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
print(f"  D = {D.flatten().round(6)}")
print("="*50)