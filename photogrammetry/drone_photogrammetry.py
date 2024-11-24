import numpy as np
import cv2 as cv

def calibrate_camera(grid_x, grid_y, size_x, size_y, images):
    """
    Calibrate camera using chessboard images.
    
    Parameters:
    grid_x (int): Number of inner corners along x direction
    grid_y (int): Number of inner corners along y direction
    size_x (float): Physical size of the grid in x direction (per square)
    size_y (float): Physical size of the grid in y direction (per square)
    images (list): List of images as numpy arrays
    
    Returns:
    tuple: (ret, mtx, dist, rvecs, tvecs)
        ret: RMS re-projection error
        mtx: Camera matrix (3x3)
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Termination criteria for cornerSubPix
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    grid_size = (grid_x, grid_y)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    
    # Create grid of points with actual physical dimensions
    x_coords = np.linspace(0, size_x * (grid_x - 1), grid_x)
    y_coords = np.linspace(0, size_y * (grid_y - 1), grid_y)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    objp[:,:2] = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    
    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    for img in images:
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, grid_size, None)
        
        if ret:
            objpoints.append(objp)
            # Refine corners to sub-pixel accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
    # Perform camera calibration
    if len(objpoints) > 0:
        return cv.calibrateCamera(
            objpoints,
            imgpoints,
            gray.shape[::-1],  # image size (width, height)
            None,
            None
        )
    else:
        raise ValueError("No valid calibration patterns found in the provided images")

class DroneCamera:
    def __init__(self):
        """Initialize DroneCamera with empty state."""
        self.R_drone_to_camera = None
        self.mtx = None
        self.dist = None
        self.img_x = None
        self.img_y = None
    
    @staticmethod
    def rpy_forward_matrix(roll, pitch, yaw):
        """
        Calculate rotation matrix from roll, pitch, yaw angles (in degrees).
        Returns R such that R[x y z]^T maps world coordinates to drone coordinates.
        
        Parameters:
        roll (float): Roll angle in degrees
        pitch (float): Pitch angle in degrees
        yaw (float): Yaw angle in degrees
        
        Returns:
        np.ndarray: 3x3 rotation matrix
        """
        # Convert to radians and reverse direction (since it's a passive transform)
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
        
        # Roll matrix
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch matrix
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw matrix
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,                      0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        R = R.transpose() # We need the passive transform, not the active
        return R
    
    def calibrate(self, roll, pitch, yaw, image, grid_x, grid_y, size_x, size_y):
        """
        Calibrate camera using a single image and drone orientation.
        
        Parameters:
        roll (float): Roll angle in degrees
        pitch (float): Pitch angle in degrees
        yaw (float): Yaw angle in degrees
        image (np.ndarray): Calibration image
        grid_x (int): Number of inner corners along x direction
        grid_y (int): Number of inner corners along y direction
        size_x (float): Physical size of the grid in x direction (per square)
        size_y (float): Physical size of the grid in y direction (per square)
        """
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(
            grid_x=grid_x,
            grid_y=grid_y,
            size_x=size_x,
            size_y=size_y,
            images=[image]
        )
        
        self.img_x, self.img_y = image.shape[0], image.shape[1]
        
        # Store intrinsic parameters
        self.mtx = mtx
        self.dist = dist
        
        # Convert rotation vector to matrix
        R, _ = cv.Rodrigues(rvecs[0])
        
        # Determine if checkerboard was rotated 180 degrees
        angle = np.linalg.norm(rvecs[0])
        if np.abs(angle / np.pi * 180 - 180) < 30:
            R_cv_to_local = np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, -1]
            ])
        else:
            R_cv_to_local = np.array([
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, -1]
            ])
        
        # Calculate drone to local transformation
        R_local_to_drone = self.rpy_forward_matrix(roll, pitch, yaw)
        
        # Calculate final transformation
        self.R_drone_to_camera = R @ R_cv_to_local.T @ R_local_to_drone.T
        
        return ret
    
    def project_point(self, point_drone):
        """
        Project a 3D point in drone coordinates to image coordinates.
        
        Parameters:
        point_drone (np.ndarray): 3D point in drone coordinates
        
        Returns:
        np.ndarray: 2D point in image coordinates (pixels)
        """
        if self.R_drone_to_camera is None or self.mtx is None or self.dist is None:
            raise ValueError("Camera not calibrated")
        point_2d, _ = cv.projectPoints(
            np.array([point_drone]), 
            self.R_drone_to_camera,
            np.zeros(3), # no translation
            self.mtx,
            self.dist
        )
        
        return point_2d[0][0]
    
    def unproject_point(self, pixel_coords):
        """
        Unproject a 2D point in image coordinates to a ray in drone coordinates.
        
        Parameters:
        pixel_coords (np.ndarray): 2D point in image coordinates (pixels)
        
        Returns:
        np.ndarray: 3D ray direction in drone coordinates (normalized)
        """
        if self.R_drone_to_camera is None or self.mtx is None or self.dist is None:
            raise ValueError("Camera not calibrated")
        
        # Undistort point
        normalized_coords = cv.undistortPoints(
            np.array([[pixel_coords]]), 
            self.mtx,
            self.dist
        )
        
        # Create ray in camera coordinates
        ray_camera = np.array([normalized_coords[0][0][0], 
                             normalized_coords[0][0][1], 
                             1.0])
        
        # Transform ray to drone coordinates
        ray_drone = self.R_drone_to_camera.T @ ray_camera
        
        # Normalize ray
        return ray_drone / np.linalg.norm(ray_drone)
    
    def backproject_point_to_local_ray(self, pixel_coords, drone_xyz, drone_roll, drone_pitch, drone_yaw):
        """
        Back-projects a 2D point in image coordinates to a ray in local coordinates.
        """
        ray_drone = self.unproject_point(pixel_coords)
        R = self.rpy_forward_matrix(drone_roll, drone_pitch, drone_yaw)
        ray_local = R.T @ ray_drone
        ray_local = ray_local / np.linalg.norm(ray_local)
        return drone_xyz, ray_local

    def backproject_point_to_local_ground(self, pixel_coords, drone_xyz, drone_roll, drone_pitch, drone_yaw):
        """
        Back-projects a 2D point in image coordinates to a (x, y) point on the ground, 
        assuming that the ground is at z = 0.
        """
        drone_xyz, ray_local = self.backproject_point_to_local_ray(pixel_coords, drone_xyz, drone_roll, drone_pitch, drone_yaw)
        ray_length = -drone_xyz[2] / ray_local[2]
        return (drone_xyz + ray_length * ray_local)[:2]
        
    def report_state(self):
        print("IMG RESOLUTION:\n", self.img_x, self.img_y)
        print("CAMERA MATRIX:\n", self.mtx)
        print("DISTORTION:\n", self.dist)
        print("DRONE-TO-CAM FORWARD MATRIX:\n", self.R_drone_to_camera)

#### How to use this:

# 1. Create a DroneCamera object
# droneCam = DroneCamera()

# 2. Take a photo of a checkerboard calibration pattern, facing straight down.
# The checkerboard should lie flat on the ground, with the long side parallel to the local y-axis.

# grid_x, grid_y = 16, 8
# img = cv.imread('calib_5_16x8.jpg')
# drone_roll, drone_pitch, drone_yaw = 0, 0, 0
# droneCam.calibrate(roll=drone_roll, pitch=drone_pitch, yaw=drone_yaw, 
#                    image=img, 
#                    grid_x=grid_x, grid_y=grid_y, size_x=1, size_y=1);

# 3. Once the thing is calibrated, you can project points from drone coordinates to image coordinates.
# Just take a photo, simultaneously record the drone's position and roll-pitch-yaw angles, and then
# you can call the following function to get the pixel coordinates of a point in the image.
# pixel_coords=np.array([508.6, 304.7])
# drone_xyz = np.array([100, 0.0, 100.0])
# drone_roll, drone_pitch, drone_yaw = 0, -34, 0
# droneCam.backproject_point_to_local_ray(pixel_coords=pixel_coords,
#                            drone_xyz=drone_xyz,
#                            drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw)

# And you can also get the intersection of the ray with the ground plane
# droneCam.backproject_point_to_local_ground(pixel_coords=pixel_coords,
#                            drone_xyz=drone_xyz,
#                            drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw)
