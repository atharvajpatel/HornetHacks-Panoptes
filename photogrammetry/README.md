# Drone photogrammetry

## What it does

The drone photogrammetry package constructs an object `DroneCamera`, which represents the intrinsic geometry of a drone equipped with a camera. It assumes that the camera has fixed intrinsics (focal length and lens distortion), and that the camera is securely fixed on the drone (that is, its position and orientation relative to the drone is fixed).

It first performs calibration by taking a single image of a checkerboard calibration pattern with the camera pointing face-down. Once calibrated, it can use back-projection to find the position and direction of the light ray corresponding to any pixel on any image. To perform back-projection, the image must be tagged with the drone position, roll, pitch, and yaw angles, at the instance when the image is taken.

![](presentation_images/drone_rpy.jpg)

By intersecting the light ray with the local topographic map, any object of interest captured in images can be mapped to the ground. Right now, we are assuming the local terrain is flat at $z = 0$. We expect to extend this with a GIS, so that we can compute with any terrain, not just flat ones.

## How to use

1. Create a `DroneCamera` object

```python
droneCam = DroneCamera()
```

2. Take a photo of a checkerboard calibration pattern, facing straight down.
The checkerboard should lie flat on the ground, with the long side parallel to the local y-axis.

```python
grid_x, grid_y = 16, 8
img = cv.imread('calib_5_16x8.jpg')
drone_roll, drone_pitch, drone_yaw = 0, 0, 0
droneCam.calibrate(roll=drone_roll, pitch=drone_pitch, yaw=drone_yaw, 
                   image=img, 
                   grid_x=grid_x, grid_y=grid_y, size_x=1, size_y=1);
```

3. Once the thing is calibrated, you can project points from drone coordinates to image coordinates.
Just take a photo, simultaneously record the drone's position and roll-pitch-yaw angles, and then
you can call the following function to get the pixel coordinates of a point in the image.

```python
pixel_coords=np.array([508.6, 304.7])
drone_xyz = np.array([100, 0.0, 100.0])
drone_roll, drone_pitch, drone_yaw = 0, -34, 0
droneCam.backproject_point_to_local_ray(pixel_coords=pixel_coords,
                           drone_xyz=drone_xyz,
                           drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw)
```

This will return two objects: `np.array([x, y, z])`, representing the coordinates of the drone,
and `np.array([v_x, v_y, v_z])`, representing the direction of the ray passing the drone.
And you can also get the intersection of the ray with the ground plane

```python
droneCam.backproject_point_to_local_ground(pixel_coords=pixel_coords,
                           drone_xyz=drone_xyz,
                           drone_roll=drone_roll, drone_pitch=drone_pitch, drone_yaw=drone_yaw)
```

This will return a single `np.array([x, y])`, representing the point on the ground that the pixel in the image corresponds to.


## User journey

Before the first light, a drone tactical unit arrives on the front line. Daylight maneuvers has become too dangerous due to the loitering drones. Settling into a camouflaged dugout, nicknamed "Drone King's Nest", they unpack and set up the gear.

The drone technician places a tall rectangular shelf with a single opening on top. On the shelf there is a little LCD display showing three numbers "row, pitch, yaw". The technician rotates the shelf and turns some nuts until the three numbers read zeros. He calls out "all balls".

He takes out a drone from a box, places it on the top of the shelf so that the camera looks through the opening, and inserts a small cable into its side. He presses a button, a single "beep", and he pulls the cable off the drone. The drone's camera is calibrated. He repeats this for every drone in the box.

The drone operators place 4 drones on the launch pads placed outside the dugout. At the dawn of day, they take off and start following a preplanned scouting trajectory.

![](presentation_images/drone_photogrammetry.jpg)

As they survey, they send back images (tagged with their geolocations and drone pose estimates) back to the ground station server, where a server matches these images with known images of the ground to improve the geolocations and pose estimates. It then runs a neural network to find potential objects of interest, such as trucks, anti-air defense systems, etc. Each object is located on the map using drone photogrammetry.

As the server performs this task, it updates a list of objects of interest, their inferred locations, and photographs of these. This is displayed on a single unified dashboard display ("The Overseer") showing a real-time local map, with multicolored squares, triangles, circles, etc, showing threats, drones, known friendly units, suspected hostile units, etc.

![](presentation_images/battlespace_overseer_2.jpg)


There are also overlapping half-circles and polygons in warning red, marking out regions of anti-drone fire. The operator can click on any symbol and out pops the drone photos taken for that object.

![](presentation_images/battlespace_overseer.jpg)
