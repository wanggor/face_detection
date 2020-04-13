import busio
import adafruit_amg88xx
import time
import board

i2c_bus = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c_bus)
time.sleep(1)
print(amg.pixels)