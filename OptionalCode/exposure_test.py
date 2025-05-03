import os
import time
from gpiozero import Button 

## Setup 
# Button Input
button_pin = 17         # Whichever pin the button is connected to (In this case its the Pi pin 11 == GPIO pin17)
button = Button(button_pin, pull_up=True)

# Exposure and gain settings (adjust these as needed)
exposure_times = [20000, 30000, 40000, 50000, 60000]  # Microseconds (5ms to 60ms)
gains = [1.0, 1.5, 2.0, 2.5, 3.0]  # Gain levels

run_count = 1  # Tracks the number of runs

print("Waiting for button press to start capturing...")

while True:  # Keeps running until manually stopped
    button.wait_for_press()
    print(f"Button pressed! Starting capture set {run_count}...")

    start_time = time.time()  # Start timer for this set

    # Capture images for each exposure and gain combination
    for exposure in exposure_times:
        for gain in gains:
            print(f"Capturing image with Exposure: {exposure} µs, Gain: {gain}")

            # Construct the filename with run_count to differentiate sets
            filename = f"exp{exposure}_gain{gain}_run{run_count}.jpg"
            command = f"libcamera-still -t 5 --camera 0 --tuning-file /usr/share/libcamera/ipa/rpi/pisp/ov5647.json --shutter {exposure} --gain {gain} -o {filename}"
            
            os.system(command)  # Execute the command

            print(f"Image saved: {filename}")
            time.sleep(0.05)  # Small delay between captures
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Finished capture set {run_count} in {elapsed_time:.2f} seconds.")

    print(f"Finished capture set {run_count}. Press button again to start a new set.")
    run_count += 1  # Increment the run count for the next set