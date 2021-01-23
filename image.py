import time

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image

RST = 24
disp = Adafruit_SSD1306.SSD1306_128_32(rst=RST)

disp.begin()
disp.clear()
disp.display()

image = Image.open('cat.png').convert('1')

def cat():
	disp.image(image)
	disp.display()

def no_cat():
	disp.clear()
	disp.display()
