import math
import random
import time
import numpy as np
from typing import List, Optional, Tuple
from threading import Lock, Thread

class Item:
    """
    Unified class representing equipment attached to a drone with threat/detection capabilities.
    Combines relative positioning (ENU) with threat assessment radius.
    """
    def __init__(self, 
                 name: str, 
                 x: float,      # East offset from drone in meters
                 y: float,      # North offset from drone in meters
                 z: float,      # Up offset from drone in meters
                 radius: float  # Threat/detection radius in meters
    ):
        # Item identity
        self.name = name
        
        # Relative position to drone (ENU coordinates in meters)
        self.x = x
        self.y = y
        self.z = 0
        self.radius = radius
        
        # Calculated absolute position (updated when drone moves)
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.altitude: float = 0.0
        
        # Threat assessment attributes
        self.nearest_threat_distance: Optional[float] = None
        self.nearest_threat_bearing: Optional[float] = None

    def calculate_distance_and_bearing(self, target_lat: float, target_lon: float, target_alt: float) -> tuple[float, float]:
        """Calculates distance and bearing to a target from item's position."""
        # Calculate 3D distance using Euclidean distance
        delta_lat = target_lat - self.latitude
        delta_lon = target_lon - self.longitude
        delta_alt = target_alt - self.altitude
        
        total_distance = math.sqrt(
            delta_lat**2 + delta_lon**2 + delta_alt**2
        )
        
        # Calculate distance to danger zone by subtracting radius
        distance_to_danger = total_distance - self.radius
        
        # Calculate bearing
        bearing = math.degrees(math.atan2(delta_lon, delta_lat))
        bearing = (90 - bearing) % 360  # Convert to 0-360° format
        
        return distance_to_danger, bearing

    def assess_threat(self, target_lat: float, target_lon: float, target_alt: float) -> dict:
        """Assesses if a target point is within item's threat/detection radius."""
        distance_to_danger, bearing = self.calculate_distance_and_bearing(target_lat, target_lon, target_alt)
        
        return {
            "distance_to_danger": distance_to_danger,
            "bearing": bearing,
            "cardinal_direction": self.get_cardinal_direction(bearing),
            "is_in_range": distance_to_danger <= 0,
            "vertical_separation": abs(target_alt - self.altitude)
        }

    @staticmethod
    def get_cardinal_direction(bearing: float) -> str:
        """Converts bearing to cardinal direction."""
        if 337.5 <= bearing <= 360 or 0 <= bearing < 22.5:
            return "N"
        elif 22.5 <= bearing < 67.5:
            return "NE"
        elif 67.5 <= bearing < 112.5:
            return "E"
        elif 112.5 <= bearing < 157.5:
            return "SE"
        elif 157.5 <= bearing < 202.5:
            return "S"
        elif 202.5 <= bearing < 247.5:
            return "SW"
        elif 247.5 <= bearing < 292.5:
            return "W"
        elif 292.5 <= bearing < 337.5:
            return "NW"

    def __str__(self) -> str:
        return (f"{self.name}: "
                f"Position(E={self.x}m, N={self.y}m, U={self.z}m), "
                f"Global({self.latitude:.6f}°, {self.longitude:.6f}°, {self.altitude:.1f}m), "
                f"Radius={self.radius}m")

class Drone:
    # WGS84 ellipsoid constants
    EARTH_A = 6378137.0  # semi-major axis [m]
    EARTH_B = 6356752.314245  # semi-minor axis [m]
    EARTH_F = (EARTH_A - EARTH_B) / EARTH_A  # flattening
    EARTH_E2 = EARTH_F * (2 - EARTH_F)  # first eccentricity squared

    def __init__(self, name: str, latitude: float = 0.0, longitude: float = 0.0, altitude: float = 0.0):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.position_lock = Lock()
        self.items: List[Item] = []

    def add_item(self, item: Item) -> None:
        """Add an item to the drone's inventory and calculate its global position."""
        self.items.append(item)
        self.update_item_positions()

    def enu_to_ecef(self, e: float, n: float, u: float) -> Tuple[float, float, float]:
        """Convert ENU coordinates to ECEF coordinates relative to drone position."""
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)

        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        R = np.array([
            [-sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon],
            [cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon],
            [0, cos_lat, sin_lat]
        ])

        enu = np.array([e, n, u])
        xyz_rel = R.T @ enu

        N = self.EARTH_A / math.sqrt(1 - self.EARTH_E2 * sin_lat * sin_lat)
        x0 = (N + self.altitude) * cos_lat * cos_lon
        y0 = (N + self.altitude) * cos_lat * sin_lon
        z0 = (N * (1 - self.EARTH_E2) + self.altitude) * sin_lat

        x = x0 + xyz_rel[0]
        y = y0 + xyz_rel[1]
        z = z0 + xyz_rel[2]

        return x, y, z

    def ecef_to_geodetic(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert ECEF coordinates to geodetic coordinates."""
        ep2 = (self.EARTH_A**2 - self.EARTH_B**2) / self.EARTH_B**2
        r = math.sqrt(x*x + y*y)
        E2 = self.EARTH_A**2 - self.EARTH_B**2
        F = 54 * self.EARTH_B**2 * z*z
        G = r*r + (1 - self.EARTH_E2)*z*z - self.EARTH_E2*E2
        c = (self.EARTH_E2*self.EARTH_E2*F*r*r)/(G*G*G)
        s = (1 + c + math.sqrt(c*c + 2*c))**(1/3)
        P = F/(3*(s + 1/s + 1)**2*G*G)
        Q = math.sqrt(1 + 2*self.EARTH_E2*self.EARTH_E2*P)
        r0 = -(P*self.EARTH_E2*r)/(1 + Q) + math.sqrt(
            (self.EARTH_A**2/2)*(1 + 1/Q) - 
            (P*(1 - self.EARTH_E2)*z*z)/(Q*(1 + Q)) - 
            P*r*r/2
        )
        U = math.sqrt((r - self.EARTH_E2*r0)**2 + z*z)
        V = math.sqrt((r - self.EARTH_E2*r0)**2 + (1 - self.EARTH_E2)*z*z)
        z0 = (self.EARTH_B**2*z)/(self.EARTH_A*V)

        lat = math.degrees(math.atan((z + ep2*z0)/r))
        lon = math.degrees(math.atan2(y, x))
        h = U*(1 - self.EARTH_B**2/(self.EARTH_A*V))

        return lat, lon, h

    def update_item_positions(self) -> None:
        """Update global positions of all items based on drone's current position."""
        for item in self.items:
            # Convert item's local ENU coordinates to ECEF
            x, y, z = self.enu_to_ecef(item.x, item.y, item.z)
            # Convert ECEF coordinates back to geodetic
            item.latitude, item.longitude, item.altitude = self.ecef_to_geodetic(x, y, z)

    def update_position(self, new_lat: float, new_lon: float, new_alt: float = 0.0) -> None:
        """Updates the drone's position and all attached items in a thread-safe manner."""
        with self.position_lock:
            self.latitude = new_lat
            self.longitude = new_lon
            self.altitude = new_alt
            self.update_item_positions()

    def __str__(self) -> str:
        return f"Drone {self.name} at ({self.latitude:.6f}, {self.longitude:.6f}, {self.altitude:.1f}m) with {len(self.items)} items"

class DefenseMap:
    def __init__(self):
        self.drones: List[Drone] = []

    def add_drone(self, drone: Drone) -> None:
        """Adds a new drone to the map."""
        self.drones.append(drone)
        print(f"Added new drone: {drone}")

    def simulate_movement(self, duration: int = 60) -> None:
        """Simulates movement of all drones and updates threat assessments."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            for drone in self.drones:
                # Simulate random movement
                new_lat = random.uniform(-90, 90)
                new_lon = random.uniform(-180, 180)
                new_alt = random.uniform(0, 1000)
                drone.update_position(new_lat, new_lon, new_alt)
                
                # Print current status
                print(f"\n{'='*80}")
                print(f"Drone Status: {drone}")
                
                # Print item positions and threat assessments
                print("\nItems and Threats:")
                for item in drone.items:
                    print(f"\n  {item}")
                    
                    # Assess threats for this item against other drones' items
                    for other_drone in self.drones:
                        if other_drone != drone:
                            for other_item in other_drone.items:
                                assessment = item.assess_threat(
                                    other_item.latitude,
                                    other_item.longitude,
                                    other_item.altitude
                                )
                                print(f"    vs {other_drone.name}'s {other_item.name}:")
                                print(f"      Distance: {assessment['distance_to_danger']:.1f}m")
                                print(f"      Bearing: {assessment['bearing']:.1f}° ({assessment['cardinal_direction']})")
                                print(f"      Status: {'IN RANGE!' if assessment['is_in_range'] else 'Out of range'}")
                                print(f"      Vertical Sep: {assessment['vertical_separation']:.1f}m")
                
                print(f"{'='*80}\n")
            
            time.sleep(5)

def create_sample_items(drone: Drone) -> None:
    """Creates a set of sample items attached to the drone."""
    items = [
        # Sensors with detection ranges
        Item("Radar Array", 0.0, 0.0, 0.5, 1000.0),      # Top-mounted radar with 1km range
        Item("Forward Camera", 0.3, 0.0, -0.1, 50.0),    # Forward camera with 50m range
        Item("Left Wing Sensor", -0.5, 0.0, 0.0, 100.0), # Left sensor with 100m range
        Item("Right Wing Sensor", 0.5, 0.0, 0.0, 100.0), # Right sensor with 100m range
        Item("Bottom Scanner", 0.0, 0.0, -0.3, 75.0),    # Bottom scanner with 75m range
        Item("Signal Jammer", 0.0, 0.2, 0.0, 250.0),     # Signal jammer with 250m effect radius
    ]
    
    for item in items:
        drone.add_item(item)
        print(f"Added {item.name} to {drone.name}")

def main():
    # Create defense map
    defense_map = DefenseMap()
    
    # Create and initialize drones with items
    drone1 = Drone("Recon-1", 34.0, -118.0, 100.0)  # East of LA
    drone2 = Drone("Cargo-2", 34.1, -118.2, 150.0)  # North of LA
    
    # Add different sets of items to each drone
    create_sample_items(drone1)
    create_sample_items(drone2)
    
    # Add drones to defense map
    defense_map.add_drone(drone1)
    defense_map.add_drone(drone2)
    
    # Start simulation in a separate thread
    print("\nStarting simulation...")
    simulation_thread = Thread(target=defense_map.simulate_movement)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    try:
        # Run for 60 seconds
        time.sleep(60)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

if __name__ == "__main__":
    main()