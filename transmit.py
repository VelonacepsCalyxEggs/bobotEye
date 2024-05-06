import serial 
import time 
arduino = serial.Serial(port='COM6', baudrate=1000000, timeout=.1) 
def send(x): 
	arduino.write(bytes(x, 'utf-8')) 
	time.sleep(0.05) 
	data = arduino.readline() 
	print(data)
	#2


