## Test to Check buttoin connection and functionality

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)  # or GPIO.BCM
button_pin = 11  # Physical pin 11, GPIO 17
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

input_state = GPIO.input(button_pin)
print("Button state: ", input_state)
GPIO.cleanup()
