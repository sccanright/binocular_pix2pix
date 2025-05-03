## Test gpiozero pins
from gpiozero import Button
from signal import pause  # To keep the program running

# Pin where the button is connected (e.g., GPIO pin 17)
button_pin = 17  
button = Button(button_pin)

# Define actions when the button is pressed and released
def on_button_pressed():
    print("Button pressed!")

def on_button_released():
    print("Button released!")

# Connect the button actions
button.when_pressed = on_button_pressed
button.when_released = on_button_released

# Keep the program running to detect button presses
print("Press the button to test it!")
pause()
