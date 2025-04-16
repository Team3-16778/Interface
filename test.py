import serial
import time

port = '/dev/ttyACM1'

ser = serial.Serial(port, 9600, timeout=1)

command = "ROTATE 100 10"

while True:
    ser.write(command.encode())
    time.sleep(2)
    response = ser.readline().decode().strip()
    print(response)

ser.close()
