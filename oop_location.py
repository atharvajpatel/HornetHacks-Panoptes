import math
import random
import time
from typing import List, Optional
from threading import Lock

class AirDefenseSystem:
    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.x = x  # Local Cartesian x coordinate
        self.y = y  # Local Cartesian y coordinate
        self.latitude = None  # Will be calculated based on drone's position
        self.longitude = None  # Will be calculated based on drone's position
        self.radius = 25.0  # Default radius, could be made configurable

    def __str__(self) -> str:
        if self.latitude is None or self.longitude is None:
            return f"{self.name} at local coordinates ({self.x}, {self.y}), " + \
                   f"global coordinates not yet calculated, " + \
                   f"with radius {self.radius}"
        return f"{self.name} at local coordinates ({self.x}, {self.y}), " + \
               f"global coordinates ({self.latitude:.6f}, {self.longitude:.6f}) " + \
               f"with radius {self.radius}"

    def update_global_coordinates(self, drone_lat: float, drone_lon: float):
        """
        Calculate global coordinates (lat/lon) based on local Cartesian coordinates
        relative to the drone's position.
        
        Uses the haversine formula in reverse to calculate new coordinates.
        """
        # Earth's radius in the same units as x and y (assuming kilometers)
        EARTH_RADIUS = 6371.0
        
        # Convert x, y to distance and bearing
        distance = math.sqrt(self.x**2 + self.y**2)
        # Calculate bearing (clockwise from north)
        bearing = math.degrees(math.atan2(self.x, self.y)) % 360
        
        # Convert drone's lat/lon to radians
        lat1 = math.radians(drone_lat)
        lon1 = math.radians(drone_lon)
        
        # Convert bearing to radians
        bearing_rad = math.radians(bearing)
        
        # Convert distance to angular distance
        angular_distance = distance / EARTH_RADIUS
        
        # Calculate new latitude
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance) +
            math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        # Calculate new longitude
        lon2 = lon1 + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1),
            math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2)
        )
        
        # Convert back to degrees
        self.latitude = math.degrees(lat2)
        self.longitude = math.degrees(lon2)

class Drone:
    def __init__(self, name: str, latitude: float = 0.0, longitude: float = 0.0):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.position_lock = Lock()
        self.bearing_to_nearest: Optional[float] = None
        self.nearest_ads: Optional[AirDefenseSystem] = None
        self.distance_to_danger: Optional[float] = None
        # Local Cartesian coordinates are always (0,0) for the drone
        self.x = 0.0
        self.y = 0.0

    def update_position(self, new_lat: float, new_lon: float) -> None:
        """Updates the drone's position in a thread-safe manner."""
        with self.position_lock:
            self.latitude = new_lat
            self.longitude = new_lon

    def _calculate_distance_and_bearing(self, ads: AirDefenseSystem) -> tuple[float, float]:
        """Calculates distance to danger zone edge and bearing to an ADS using local coordinates."""
        # Calculate total distance using Cartesian coordinates
        total_distance = math.sqrt(ads.x**2 + ads.y**2)
        
        # Calculate distance to danger zone by subtracting radius
        distance_to_danger = total_distance - ads.radius
        
        # Calculate bearing using arctangent of local coordinates
        bearing = math.degrees(math.atan2(ads.x, ads.y)) % 360
        
        return distance_to_danger, bearing

    def assess_threats(self, defense_systems: List[AirDefenseSystem]) -> dict:
        """Assesses all threats and updates drone's bearing to nearest threat."""
        threat_assessment = {}
        closest_distance = float('inf')
        
        for ads in defense_systems:
            # Update ADS global coordinates based on drone's position
            ads.update_global_coordinates(self.latitude, self.longitude)
            
            distance_to_danger, bearing = self._calculate_distance_and_bearing(ads)
            
            # Update nearest ADS if this is the closest one
            if distance_to_danger < closest_distance:
                closest_distance = distance_to_danger
                self.bearing_to_nearest = bearing
                self.nearest_ads = ads
                self.distance_to_danger = distance_to_danger
            
            threat_assessment[ads.name] = {
                "distance_to_danger": distance_to_danger,
                "bearing": bearing,
                "status": True if distance_to_danger < 0 else False,
                "ads_latitude": ads.latitude,
                "ads_longitude": ads.longitude,
                "local_x": ads.x,
                "local_y": ads.y
            }
        
        return dict(sorted(
            threat_assessment.items(),
            key=lambda x: x[1]["distance_to_danger"]
        ))

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

class DefenseMap:
    def __init__(self):
        self.defense_systems: List[AirDefenseSystem] = []
        self.drones: List[Drone] = []

    def add_defense_system(self, name: str, x: float, y: float) -> None:
        """Adds a new air defense system to the map using local coordinates."""
        ads = AirDefenseSystem(name, x, y)
        self.defense_systems.append(ads)
        print(f"Added new defense system: {ads}")

    def add_drone(self, name: str, latitude: float = 0.0, longitude: float = 0.0) -> None:
        """Adds a new drone to the map."""
        drone = Drone(name, latitude, longitude)
        self.drones.append(drone)
        print(f"Added new drone: {drone.name} at ({latitude}, {longitude})")

    def simulate_movement(self) -> None:
        """Simulates movement of all drones and updates threat assessments."""
        while True:
            for drone in self.drones:
                # Simulate random movement
                new_lat = random.uniform(-90, 90)
                new_lon = random.uniform(-180, 180)
                drone.update_position(new_lat, new_lon)
                
                # Get threat assessment
                threats = drone.assess_threats(self.defense_systems)
                
                # Print current status
                print(f"\nDrone {drone.name} Position: ({drone.latitude:.6f}, {drone.longitude:.6f})")
                print("\nThreat Assessment:")
                for threat_name, data in threats.items():
                    print(f"{threat_name}:")
                    print(f"  Local Coordinates (x,y): ({data['local_x']:.2f}, {data['local_y']:.2f})")
                    print(f"  Calculated Global Position: ({data['ads_latitude']:.6f}, {data['ads_longitude']:.6f})")
                    print(f"  Distance To Danger Zone: {data['distance_to_danger']:.2f} Units")
                    print(f"  Bearing: {data['bearing']:.2f}° ({Drone.get_cardinal_direction(data['bearing'])})")
                    print("  Status: INSIDE DANGER ZONE!" if data['status'] else "  Status: OUTSIDE DANGER ZONE!")
            
            time.sleep(5)

def main():
    # Create defense map
    defense_map = DefenseMap()
    
    # Add initial defense systems with local Cartesian coordinates
    defense_map.add_defense_system("SAM Site Alpha", 10.0, 15.0)
    defense_map.add_defense_system("SAM Site Beta", -5.0, 20.0)
    defense_map.add_defense_system("SAM Site Gamma", 8.0, -12.0)
    
    # Add drones
    defense_map.add_drone("Drone-1")
    defense_map.add_drone("Drone-2", 34.0, -118.0)
    
    # Start simulation
    defense_map.simulate_movement()

if __name__ == "__main__":
    main()