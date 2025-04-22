import serial
import time

port = '/dev/ttyACM0'

ser = serial.Serial(port, 9600,timeout=1)

command = "HOME\n"


ser.write(command.encode())
time.sleep(2)
response = ser.readline().decode().strip()
print(response)

ser.close()
