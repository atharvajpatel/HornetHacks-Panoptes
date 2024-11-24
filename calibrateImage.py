from drone_photogrammetry import DroneCamera
import cv2 as cv
import pickle

drone = DroneCamera()

def calibrate(path):
    grid_x, grid_y = 16, 8
    img = cv.imread(path)
    drone_roll, drone_pitch, drone_yaw = 0, 0, 0 #Placeholder for now
    drone.calibrate(roll=drone_roll, pitch=drone_pitch, yaw=drone_yaw, image=img, grid_x=grid_x, grid_y=grid_y, size_x=1, size_y=1)


path = 'calibrator.jpg'
calibrate(path)
drone.report_state()

# Pickle the drone object
with open('drone.pkl', 'wb') as file:
    pickle.dump(drone, file)
print("\nDrone object has been pickled to 'drone.pkl'")

# Unpickle the drone object
def unpickle():
    with open('drone.pkl', 'rb') as file:
        loaded_drone = pickle.load(file)
    print("\nUnpickled drone:", loaded_drone)
    return loaded_drone





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

# This will return two objects: np.array([x, y, z]), representing the coordinates of the drone,
# and a numpy array np.array([v_x, v_y, v_z]), representing the direction of the ray passing the drone.
# (array([100.   0. 100.]), array([ 0.70127654, -0.04217534, -0.71164068]))

# And you can also get the intersection of the ray with the ground plane
# droneCam.backproject_point_to_local_ground(pixel_coords=pixel_coords,
#                            drone_xyz=drone_xyz,
#                            drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw)
# This will return a single np.array([x, y]), representing the point on the ground 
# that the pixel in the image corresponds to.
