import serial

class ReceiveCoords:
    def __init__(self):
        self.port = serial.Serial("10.33.0.210", baudrate=115200, timeout=3.0)
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.9

    def get_coords(self):
        rcv = self.port.read()
        # port.write("\r\nYou sent:" + repr(rcv))
        self.lat, self.lon, self.alt, self.roll, self.pitch, self.yaw = repr(rcv).split(',') # lat, lon, alt, roll, pitch, yaw
        return [self.lat, self.lon, self.alt, self.roll, self.pitch, self.yaw]

    def get_lat_lon_yaw(self):
        return [self.lat, self.lon, self.yaw]
