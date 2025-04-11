import serial
import time

port = '/dev/ttyACM1'

ser = serial.Serial(port, 9600, timeout=1)

command = "GOTO 100 100 100"

while True:
    ser.write(command.encode())
    time.sleep(2)
    response = ser.readline().decode().strip()
    print(response)

ser.close()
