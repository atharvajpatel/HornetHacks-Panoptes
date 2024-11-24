import socket
import json
import time

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 12345))

# Example position update
position = [37.7749, -122.4194, 45.0]  # [latitude, longitude, yaw]
message = json.dumps(position) + '\n'
client.send(message.encode('utf-8'))