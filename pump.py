import RPi.GPIO as GPIO
import time

pin = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)

def switch_pump(state):
    GPIO.output(pin, state)


