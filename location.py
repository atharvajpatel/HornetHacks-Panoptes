import math
from typing import Dict, List, Tuple
from threading import Lock
import time
import random

# Global variable to store drone's current position with thread-safe access
drone_position = {"latitude": 0.0, "longitude": 0.0}
position_lock = Lock()

# Dictionary to store air defense systems
air_defense_systems = {
    "SAM_1": {"name": "SAM Site Alpha", "latitude": 34.0522, "longitude": -118.2437, "radius": 25},
    "SAM_2": {"name": "SAM Site Beta", "latitude": 34.1478, "longitude": -118.1445, "radius": 30},
    "SAM_3": {"name": "SAM Site Gamma", "latitude": 34.0331, "longitude": -118.3561, "radius": 20}
}

def update_drone_position(new_lat: float, new_lon: float) -> None:
    """
    Updates the drone's position in a thread-safe manner.
    """
    global drone_position
    with position_lock:
        drone_position["latitude"] = new_lat
        drone_position["longitude"] = new_lon

def calculate_distance_and_bearing(lat1: float, lon1: float, lat2: float, lon2: float, radius: float) -> Tuple[float, float]:
    """
    Calculates distance to danger zone edge and bearing from drone to threat.
    
    Args:
        lat1, lon1: Drone position
        lat2, lon2: Threat position
        radius: Radius of the threat's danger zone
    
    Returns:
        Tuple of (distance to danger zone, bearing in degrees)
        Note: Negative distance means drone is inside danger zone
        Bearing is in degrees from North (0-360°):
        - 0° is North
        - 90° is East
        - 180° is South
        - 270° is West
    """
    # Calculate total distance between points
    total_distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    
    # Calculate distance to danger zone by subtracting radius
    distance_to_danger = total_distance - radius
    
    # Calculate bearing
    # Get the differences in coordinates
    delta_lat = lat2 - lat1  # Change in latitude
    delta_lon = lon2 - lon1  # Change in longitude
    
    # Calculate bearing using arctangent
    bearing = math.degrees(math.atan2(delta_lon, delta_lat))
    
    # Convert bearing to 0-360° format (clockwise from North)
    bearing = (90 - bearing) % 360
    
    return distance_to_danger, bearing

def assess_threats() -> Dict[str, Dict[str, float]]:
    """
    Calculates distance to danger zone and bearing for all threats, 
    returns sorted dictionary by closest threat.
    """
    global drone_position, air_defense_systems
    threat_assessment = {}
    
    with position_lock:
        current_lat = drone_position["latitude"]
        current_lon = drone_position["longitude"]
    
    for system_id, system in air_defense_systems.items():
        distance_to_danger, bearing = calculate_distance_and_bearing(
            current_lat, current_lon,
            system["latitude"], system["longitude"],
            system["radius"]
        )
        
        threat_assessment[system["name"]] = {
            "distance_to_danger": distance_to_danger,
            "bearing": bearing,
            'status': "INSIDE DANGER ZONE!" if distance_to_danger < 0 else "OUTSIDE DANGER ZONE!"
        }
    
    # Sort by distance_to_danger
    return dict(sorted(
        threat_assessment.items(),
        key=lambda x: x[1]["distance_to_danger"]
    ))

def simulate_drone_movement():
    """
    Simulates drone movement and threat assessment.
    """
    while True:
        # Simulate receiving new position data
        update_drone_position(random.uniform(-90, 90), random.uniform(-180, 180))
        
        # Get threat assessment
        threats = assess_threats()
        print("\nCurrent Drone Position:", drone_position)
        print("\nThreat Assessment:")
        for threat_name, data in threats.items():
            print(f"{threat_name}:")
            print(f"  Distance to danger zone: {data['distance_to_danger']:.2f} units")
            print(f"  Bearing: {data['bearing']:.2f}° ({get_cardinal_direction(data['bearing'])})")
            print(f"  Status: {data['status']}")
        
        time.sleep(5)

def get_cardinal_direction(bearing: float) -> str:
    """
    Converts bearing to cardinal direction for better understanding.
    """
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


if __name__ == "__main__":
    simulate_drone_movement()