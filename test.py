import socket
import json
import receive_coords

def send_drone_data(host, port, data):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        
        # Send drone data
        message = json.dumps(data) + "\n"  # Ensure each JSON object ends with a newline
        client_socket.sendall(message.encode('utf-8'))
        print(f"Sent: {message}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        client_socket.close()
        print("Connection closed.")

# Example usage
if __name__ == "__main__":
    host = '10.33.0.210'  # Replace with your server's IP address or hostname
    port = 5000  # Replace with your server's port if different
    coords = receive_coords.ReceiveCoords()

    # Define some sample drone data (latitude, longitude, yaw)
    while(True):
        coords.get_coords()
        send_drone_data(host, port, coords.get_lat_lon_yaw)