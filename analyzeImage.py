import pickle
import numpy as np
from inferenceMidpixel import MilitaryDetector

with open('drone.pkl', 'rb') as file:
    drone = pickle.load(file)
    print("\nUnpickled drone:", drone)

drone.report_state()

path = 'aerial soldier.jpg'
detector = MilitaryDetector(model_path='yolov5n.pt', input_file_path=path, conf_threshold=0.15)
detector.detect(save_path="aerial-soldiers2-box.jpg")

all_coordinates = detector.pixel_coordinates.getDict()

print(all_coordinates)

def get_first_coordinate(coordinates_dict):
    """Extract the first available coordinate from any class."""
    for class_name, coords_list in coordinates_dict.items():
        if coords_list:  # If this class has any coordinates
            coord = coords_list[0]  # Get first coordinate tuple
            print(f"\nFound first coordinate in class: {class_name}")
            return np.array(coord)
    
    return None  # Return None if no coordinates found at all

# Get the first available coordinate
pixel_coords = get_first_coordinate(all_coordinates)

if pixel_coords is not None:
    print(f"First available coordinate as numpy array: {pixel_coords}")
    print(f"x coordinate: {pixel_coords[0]}")
    print(f"y coordinate: {pixel_coords[1]}")
else:
    print("No coordinates found in any class")


# Convert pixel coordinates to float type
pixel_coords = np.array(pixel_coords, dtype=float)

# Set up drone parameters
drone_xyz = np.array([100, 0.0, 100.0])
drone_roll, drone_pitch, drone_yaw = 0, -34, 0

# Get backprojection result
position = drone.backproject_point_to_local_ground(pixel_coords=pixel_coords,
                           drone_xyz=drone_xyz,
                           drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw)

# The position array is our XYZ Cartesian point
xyz_cartesian = position



"""
# 3. Once the thing is calibrated, you can project points from drone coordinates to image coordinates.
# Just take a photo, simultaneously record the drone's position and roll-pitch-yaw angles, and then
# you can call the following function to get the pixel coordinates of a point in the image.
pixel_coords=np.array(pixel_coords, dtype=float)
drone_xyz = np.array([100, 0.0, 100.0])
drone_roll, drone_pitch, drone_yaw = 0, -34, 0
print(drone.backproject_point_to_local_ray(pixel_coords=pixel_coords,
                           drone_xyz=drone_xyz,
                           drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw))

# This will return two objects: np.array([x, y, z]), representing the coordinates of the drone,
# and a numpy array np.array([v_x, v_y, v_z]), representing the direction of the ray passing the drone.
# (array([100.   0. 100.]), array([ 0.70127654, -0.04217534, -0.71164068]))

# And you can also get the intersection of the ray with the ground plane
drone.backproject_point_to_local_ground(pixel_coords=pixel_coords,
                           drone_xyz=drone_xyz,
                           drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw)

# This will return a single np.array([x, y]), representing the point on the ground 
# that the pixel in the image corresponds to.

"""
