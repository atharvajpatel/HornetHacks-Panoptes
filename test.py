import socket
import json

def send_drone_data(host, port, drone_data):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        
        # Send drone data
        for data in drone_data:
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
    host = '10.33.1.104'  # Replace with your server's IP address or hostname
    port = 12345  # Replace with your server's port if different

    # Define some sample drone data (latitude, longitude, yaw)
    drone_data = [
        [34.0, -118.0, 45.0],
        [35.0, -119.0, 90.0],
        [36.0, -120.0, 135.0]
    ]

    send_drone_data(host, port, drone_data)