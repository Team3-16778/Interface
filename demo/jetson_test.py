import serial
import time

# Open serial connections
try:
    arduino1 = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
except:
    arduino1 = None
    print("Arduino 1 not found.")

try:
    arduino2 = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
except:
    arduino2 = None
    print("Arduino 2 not found.")

time.sleep(2)  # Allow Arduino to reset

def send_and_receive(arduino, name, command):
    if arduino:
        arduino.write((command + '\n').encode())
        time.sleep(0.1)
        response = arduino.readline().decode().strip()
        print(f"{name}: Sent '{command}', Received: '{response}'")

while True:
    send_and_receive(arduino1, "Arduino 1", "LED_ON")
    send_and_receive(arduino2, "Arduino 2", "LED_ON")
    time.sleep(3)
