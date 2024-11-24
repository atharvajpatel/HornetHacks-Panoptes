import math
import time
from typing import List, Optional
from threading import Lock
import socket
import json
import threading
import argparse
import ssl

class AirDefenseSystem:
    def __init__(self, name: str, x: float, y: float, yaw: float):
        self.name = name
        # Store original coordinates
        self.original_x = x
        self.original_y = y
        self.yaw = yaw  # Store yaw angle in radians
        
        # Calculate rotated coordinates using rotation matrix
        self.x = x * math.cos(yaw) - y * math.sin(yaw)
        self.y = x * math.sin(yaw) + y * math.cos(yaw)
        
        self.latitude = None
        self.longitude = None
        self.radius = 25.0

    def __str__(self) -> str:
        if self.latitude is None or self.longitude is None:
            return (f"{self.name} at original coordinates ({self.original_x}, {self.original_y}), "
                   f"rotated coordinates ({self.x:.2f}, {self.y:.2f}), "
                   f"yaw {math.degrees(self.yaw):.2f}°, "
                   f"global coordinates not yet calculated, "
                   f"with radius {self.radius}")
        return (f"{self.name} at original coordinates ({self.original_x}, {self.original_y}), "
               f"rotated coordinates ({self.x:.2f}, {self.y:.2f}), "
               f"yaw {math.degrees(self.yaw):.2f}°, "
               f"global coordinates ({self.latitude:.6f}, {self.longitude:.6f}) "
               f"with radius {self.radius}")

    def update_global_coordinates(self, drone_lat: float, drone_lon: float):
        """
        Calculate global coordinates (lat/lon) based on rotated local Cartesian coordinates
        relative to the drone's position.
        
        Uses the haversine formula in reverse to calculate new coordinates.
        """
        # Earth's radius in the same units as x and y (assuming kilometers)
        EARTH_RADIUS = 6371.0
        
        # Use rotated coordinates for calculations
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
    def __init__(self, name: str, latitude: float = 0.0, longitude: float = 0.0, yaw: float = 0.0):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.yaw = yaw
        self.position_lock = Lock()
        self.bearing_to_nearest: Optional[float] = None
        self.nearest_ads: Optional[AirDefenseSystem] = None
        self.distance_to_danger: Optional[float] = None
        self.x = 0.0
        self.y = 0.0

    def update_position(self, new_lat: float, new_lon: float, new_yaw: float) -> None:
        """Updates the drone's position in a thread-safe manner."""
        with self.position_lock:
            self.latitude = new_lat
            self.longitude = new_lon
            self.yaw = new_yaw

    def _calculate_distance_and_bearing(self, ads: AirDefenseSystem) -> tuple[float, float]:
        """Calculates distance to danger zone edge and bearing using rotated coordinates."""
        # Calculate total distance using rotated Cartesian coordinates
        total_distance = math.sqrt(ads.x**2 + ads.y**2)
        
        # Calculate distance to danger zone by subtracting radius
        distance_to_danger = total_distance - ads.radius
        
        # Calculate bearing using arctangent of rotated local coordinates
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
                "original_x": ads.original_x,
                "original_y": ads.original_y,
                "rotated_x": ads.x,
                "rotated_y": ads.y,
                "yaw_degrees": math.degrees(ads.yaw)
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
    def __init__(self, host='0.0.0.0', port=12345):
        self.defense_systems: List[AirDefenseSystem] = []
        self.drones: List[Drone] = []
        self.socket_host = host
        self.socket_port = port
        self.running = True

    def add_defense_system(self, name: str, x: float, y: float, yaw: float) -> None:
        """Adds a new air defense system to the map using local coordinates and yaw angle."""
        ads = AirDefenseSystem(name, x, y, yaw)
        self.defense_systems.append(ads)
        print(f"Added new defense system: {ads}")

    def add_drone(self, name: str, latitude: float = 0.0, longitude: float = 0.0) -> None:
        """Adds a new drone to the map."""
        drone = Drone(name, latitude, longitude)
        self.drones.append(drone)
        print(f"Added new drone: {drone.name} at ({latitude}, {longitude})")

    def start_socket_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.socket_host, self.socket_port))
        server_socket.listen(5)
        print(f"Socket server listening on {self.socket_host}:{self.socket_port}")
        
        while self.running:
            try:
                client_socket, address = server_socket.accept()
                print(f"Connection from {address}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.start()
            except Exception as e:
                print(f"Error accepting connection: {e}")

    def handle_client(self, client_socket):
        """Handles incoming data from a connected client."""
        buffer = ""
        while self.running:
            try:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break

                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        # Parse the JSON data
                        position_data = json.loads(line)
                        
                        # Expect [latitude, longitude, yaw]
                        if len(position_data) == 3:
                            lat, lon, yaw = position_data
                            
                            # Update all drones (you might want to modify this to update specific drones)
                            for drone in self.drones:
                                drone.update_position(lat, lon, yaw)
                                
                                # Get and print threat assessment
                                threats = drone.assess_threats(self.defense_systems)
                                self.print_threat_assessment(drone, threats)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
                    
            except Exception as e:
                print(f"Error handling client data: {e}")
                break

        client_socket.close()

    def print_threat_assessment(self, drone, threats):
        """Prints the current threat assessment."""
        print(f"\nDrone {drone.name} Position: ({drone.latitude:.6f}, {drone.longitude:.6f}, {drone.yaw:.2f}°)")
        print("\nThreat Assessment:")
        for threat_name, data in threats.items():
            print(f"{threat_name}:")
            print(f"  Original Coordinates (x,y): ({data['original_x']:.2f}, {data['original_y']:.2f})")
            print(f"  Rotated Coordinates (x,y): ({data['rotated_x']:.2f}, {data['rotated_y']:.2f})")
            print(f"  Yaw Angle: {data['yaw_degrees']:.2f}°")
            print(f"  Calculated Global Position: ({data['ads_latitude']:.6f}, {data['ads_longitude']:.6f})")
            print(f"  Distance To Danger Zone: {data['distance_to_danger']:.2f} Units")
            print(f"  Bearing: {data['bearing']:.2f}° ({Drone.get_cardinal_direction(data['bearing'])})")
            print("  Status: INSIDE DANGER ZONE!" if data['status'] else "  Status: OUTSIDE DANGER ZONE!")

def main():
    parser = argparse.ArgumentParser(description="Defense Map Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=12345, help='Port to listen on')
    args = parser.parse_args()

    defense_map = DefenseMap(host=args.host, port=args.port)
    
    # Add initial defense systems with local Cartesian coordinates and yaw angles
    defense_map.add_defense_system("SAM Site Alpha", 10.0, 15.0, math.radians(45))  # 45° rotation
    defense_map.add_defense_system("SAM Site Beta", -5.0, 20.0, math.radians(90))   # 90° rotation
    defense_map.add_defense_system("SAM Site Gamma", 8.0, -12.0, math.radians(30))  # 30° rotation
    
    # Add drones
    defense_map.add_drone("Drone-1")
    defense_map.add_drone("Drone-2", 34.0, -118.0)
    
    # Start the socket server
    try:
        defense_map.start_socket_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
        defense_map.running = False

if __name__ == "__main__":
    main()