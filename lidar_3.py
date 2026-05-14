import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar, RPLidarException
import open3d as o3d
import time

# ===== CONFIGURATION =====
PORT = "COM31"
MAX_FRAMES = 1500
DIST_THRESHOLD = 0.5  # ICP max correspondence distance (meters)

# ===== INIT LIDAR =====
lidar = RPLidar(PORT, baudrate=115200, timeout=3)

try:
    lidar.reset()
    time.sleep(2)
except:
    pass

if lidar._serial_port is not None:
    lidar._serial_port.reset_input_buffer()

# ===== DATA STORAGE FOR FGO =====
# We store: [timestamp, global_x, global_y, global_theta, delta_x, delta_y, delta_theta]
fgo_registry = []
current_pose = np.eye(4)
prev_pcd = None

# ===== PREPARE PLOT (FAST UPDATE MODE) =====
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
ray_line, = ax.plot([], [], color='lightblue', alpha=0.3, linewidth=0.5, zorder=1)
wall_dots = ax.scatter([], [], s=2, c='black', zorder=2)
traj_line, = ax.plot([], [], 'r-', linewidth=1.5, label="Trajectory", zorder=3)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.legend()
ax.set_title("LiDAR SLAM: Ground Tuth Data Collection")

def scan_to_pcd(scan):
    pts = []
    for (_, ang, dist) in scan:
        if dist > 0:
            r = dist / 1000.0
            a = np.deg2rad(ang)
            pts.append([r * np.cos(a), r * np.sin(a), 0])
    pcd = o3d.geometry.PointCloud()
    if len(pts) > 0:
        pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    return pcd

print("Scanning... Move robot in a circle.")

frame_count = 0
try:
    for scan in lidar.iter_scans():
        if frame_count > MAX_FRAMES:
            break
            
        pcd = scan_to_pcd(scan)
        if len(pcd.points) < 10:
            continue

        # Initial guess for ICP (Identity or prediction)
        t_guess = np.eye(4)
        
        # Relative movement (delta)
        dx, dy, dtheta = 0.0, 0.0, 0.0

        if prev_pcd is not None:
            # ICP Registration
            reg = o3d.pipelines.registration.registration_icp(
                pcd, prev_pcd, DIST_THRESHOLD, t_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Extract Delta Transformation
            rel_t = reg.transformation
            dx = rel_t[0, 3]
            dy = rel_t[1, 3]
            dtheta = np.arctan2(rel_t[1, 0], rel_t[0, 0])
            
            # Update Global Pose
            current_pose = current_pose @ rel_t

        # Extract Global x, y, theta for logging
        gx = current_pose[0, 3]
        gy = current_pose[1, 3]
        gtheta = np.arctan2(current_pose[1, 0], current_pose[0, 0])

        # Save to Registry
        fgo_registry.append([time.time(), gx, gy, gtheta, dx, dy, dtheta])

        # ===== FAST VISUALIZATION =====
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            # 1. Update Walls
            wall_dots.set_offsets(pts[:, :2])
            
            # 2. Update Laser Rays (Interleave origin and points with NaNs)
            rays_x = np.zeros(len(pts) * 3)
            rays_y = np.zeros(len(pts) * 3)
            rays_x[1::3] = pts[:, 0]
            rays_y[1::3] = pts[:, 1]
            rays_x[2::3] = np.nan
            rays_y[2::3] = np.nan
            ray_line.set_data(rays_x, rays_y)

        # 3. Update Trajectory
        traj_arr = np.array(fgo_registry)
        traj_line.set_data(traj_arr[:, 1], traj_arr[:, 2])

        # Dynamic Scaling
        ax.set_xlim(gx - 5, gx + 5)
        ax.set_ylim(gy - 5, gy + 5)
        
        plt.pause(0.001)
        
        prev_pcd = pcd
        frame_count += 1

except (RPLidarException, KeyboardInterrupt) as e:
    print(f"\nStopping: {e}")

finally:
    print("Cleaning up and saving...")
    lidar.stop()
    lidar.disconnect()

    if fgo_registry:
        data_to_save = np.array(fgo_registry)
        header = "timestamp,global_x,global_y,global_theta,delta_x,delta_y,delta_theta"
        np.savetxt("lidar_expt1.csv", data_to_save, delimiter=",", header=header)
        print(f"Success! Saved {len(data_to_save)} frames to 'lidar_expt1.csv'")
    else:
        print("No data collected.")

plt.ioff()
plt.show()