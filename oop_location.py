import math
import random
import time
from typing import List, Optional
from threading import Lock

class AirDefenseSystem:
    def __init__(self, name: str, latitude: float, longitude: float, radius: float):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.radius = radius

    def __str__(self) -> str:
        return f"{self.name} at ({self.latitude}, {self.longitude}) with radius {self.radius}"

class Drone:
    def __init__(self, name: str, latitude: float = 0.0, longitude: float = 0.0):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.position_lock = Lock()
        self.bearing_to_nearest: Optional[float] = None
        self.nearest_ads: Optional[AirDefenseSystem] = None
        self.distance_to_danger: Optional[float] = None

    def update_position(self, new_lat: float, new_lon: float) -> None:
        """Updates the drone's position in a thread-safe manner."""
        with self.position_lock:
            self.latitude = new_lat
            self.longitude = new_lon

    def _calculate_distance_and_bearing(self, ads: AirDefenseSystem) -> tuple[float, float]:
        """Calculates distance to danger zone edge and bearing to an ADS."""
        # Calculate total distance between points
        total_distance = math.sqrt(
            (ads.latitude - self.latitude)**2 + 
            (ads.longitude - self.longitude)**2
        )
        
        # Calculate distance to danger zone by subtracting radius
        distance_to_danger = total_distance - ads.radius
        
        # Calculate bearing
        delta_lat = ads.latitude - self.latitude
        delta_lon = ads.longitude - self.longitude
        
        # Calculate bearing using arctangent
        bearing = math.degrees(math.atan2(delta_lon, delta_lat))
        
        # Convert bearing to 0-360° format (clockwise from North)
        bearing = (90 - bearing) % 360
        
        return distance_to_danger, bearing

    def assess_threats(self, defense_systems: List[AirDefenseSystem]) -> dict:
        """Assesses all threats and updates drone's bearing to nearest threat."""
        threat_assessment = {}
        closest_distance = float('inf')
        
        for ads in defense_systems:
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
                "status": True if distance_to_danger < 0 else False
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

    def add_defense_system(self, name: str, latitude: float, longitude: float, radius: float) -> None:
        """Adds a new air defense system to the map."""
        ads = AirDefenseSystem(name, latitude, longitude, radius)
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
                print(f"\nDrone {drone.name} Position: ({drone.latitude:.2f}, {drone.longitude:.2f})")
                print("\nThreat Assessment:")
                for threat_name, data in threats.items():
                    print(f"{threat_name}:")
                    print(f"  Distance To Danger Zone: {data['distance_to_danger']:.2f} Units")
                    print(f"  Bearing: {data['bearing']:.2f}° ({Drone.get_cardinal_direction(data['bearing'])})")
                    print("  Status: INSIDE DANGER ZONE!" if data['status'] else "  Status: OUTSIDE DANGER ZONE!")
            
            time.sleep(5)

def main():
    # Create defense map
    defense_map = DefenseMap()
    
    # Add initial defense systems
    defense_map.add_defense_system("SAM Site Alpha", 34.0522, -118.2437, 25)
    defense_map.add_defense_system("SAM Site Beta", 34.1478, -118.1445, 30)
    defense_map.add_defense_system("SAM Site Gamma", 34.0331, -118.3561, 20)
    
    # Add drones
    defense_map.add_drone("Drone-1")
    defense_map.add_drone("Drone-2", 34.0, -118.0)
    
    # Start simulation
    defense_map.simulate_movement()

if __name__ == "__main__":
    main()