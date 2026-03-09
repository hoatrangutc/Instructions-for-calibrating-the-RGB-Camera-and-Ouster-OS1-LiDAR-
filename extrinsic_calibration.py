import numpy as np
import cv2
import glob, os

K = np.array([
    [448.9443277836273,  0.0,              328.1278074020808],
    [0.0,               449.8824080758361, 229.13027052795243],
    [0.0,               0.0,              1.0]
])
D = np.array([0.08804103349773844, -0.012905824328900526,
              -0.002744215700176299, -0.007056489723647028,
              -0.4767527602193595])

cam_files   = sorted(glob.glob("cam_frames/*.png"))
lidar_ts    = np.load("lidar_timestamps.npy")
lidar_files = sorted(glob.glob("lidar_frames/*.npy"))
cam_ts = np.array([
    int(os.path.basename(f).split("_")[1].split(".")[0])
    for f in cam_files
])
synced = []
for ct, cf in zip(cam_ts, cam_files):
    diffs   = np.abs(lidar_ts - ct)
    min_idx = np.argmin(diffs)
    if diffs[min_idx] < 100_000_000:
        synced.append((cf, lidar_files[min_idx]))

cf, lf   = synced[155] # chọn frame mẫu
img_orig = cv2.imread(cf)
pts      = np.load(lf)

# Lấy TẤT CẢ points (không filter hướng)
# Chỉ loại NaN và points quá xa
dist = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
mask = (dist > 0.1) & (dist < 15.0) & np.isfinite(pts).all(axis=1)
pts_all = pts[mask, :3]
print(f"Total points to project: {len(pts_all)}")

# Option 3: Lx→-Cz Ly→-Cx Lz→Cy
R_base = np.array([
    [ 0, -1,  0],
    [ 0,  0,  1],
    [-1,  0,  0]
], dtype=float)

tx, ty, tz = 0.0, 0.0, 0.0
rx, ry, rz = 0.0, 0.0, 0.0
step_t = 0.02
step_r = 1.0
dot_size = 2  # size của điểm

def make_R_delta(rx_deg, ry_deg, rz_deg):
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def render(R, T):
    img = img_orig.copy()
    pts_cam = (R @ pts_all.T).T + T
    valid   = pts_cam[:,2] > 0.1
    pts_img = (K @ pts_cam[valid].T).T
    pts_img = pts_img[:,:2] / pts_img[:,2:3]

    depth  = pts_cam[valid, 2]
    d_norm = np.clip(depth / 10.0, 0, 1)
    colors = cv2.applyColorMap(
        (d_norm*255).astype(np.uint8), cv2.COLORMAP_JET
    ).reshape(-1, 3)

    h, w = img.shape[:2]
    n = 0
    for p, c in zip(pts_img.astype(int), colors):
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, tuple(p), dot_size, c.tolist(), -1)
            n += 1

    # HUD
    cv2.putText(img,
        f"T=[{T[0]:.3f},{T[1]:.3f},{T[2]:.3f}]",
        (5,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    cv2.putText(img,
        f"dR=[{rx:.1f},{ry:.1f},{rz:.1f}] pts={n} step={step_t:.3f}m dot={dot_size}",
        (5,44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
    cv2.putText(img,
        "W/S:tx  A/D:ty  Q/E:tz  I/K:rx  J/L:ry  U/O:rz",
        (5,62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.putText(img,
        "F:fine  C:coarse  Z/X:dotsize  SPACE:save  ESC:quit",
        (5,78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    return img

print("Click vào cửa sổ ảnh trước khi nhấn phím!")
print("Phím: W/S tx  A/D ty  Q/E tz  I/K rx  J/L ry  U/O rz")
print("      F:fine(5mm)  C:coarse(2cm)  Z/X:dot size  SPACE:save  ESC:quit")

while True:
    R_delta = make_R_delta(rx, ry, rz)
    R_cur   = R_delta @ R_base
    T_cur   = np.array([tx, ty, tz])
    img_r   = render(R_cur, T_cur)

    cv2.imshow("Fine-tune (click window first!)", img_r)
    key = cv2.waitKey(0) & 0xFF

    if   key == 27:        break
    elif key == ord('w'):  tx += step_t
    elif key == ord('s'):  tx -= step_t
    elif key == ord('a'):  ty += step_t
    elif key == ord('d'):  ty -= step_t
    elif key == ord('q'):  tz += step_t
    elif key == ord('e'):  tz -= step_t
    elif key == ord('i'):  rx += step_r
    elif key == ord('k'):  rx -= step_r
    elif key == ord('j'):  ry += step_r
    elif key == ord('l'):  ry -= step_r
    elif key == ord('u'):  rz += step_r
    elif key == ord('o'):  rz -= step_r
    elif key == ord('f'):  step_t = 0.005
    elif key == ord('c'):  step_t = 0.02
    elif key == ord('z'):  dot_size = max(1, dot_size-1)
    elif key == ord('x'):  dot_size += 1
    elif key == ord(' '):
        np.save("R_lidar2cam.npy", R_cur)
        np.save("T_lidar2cam.npy", T_cur)
        cv2.imwrite("extrinsic_final.png", img_r)
        print(f"\n✓ SAVED!")
        print(f"R =\n{R_cur.round(6)}")
        print(f"T = {T_cur}")
        print(f"Euler delta: rx={rx:.1f}° ry={ry:.1f}° rz={rz:.1f}°")

cv2.destroyAllWindows()