import cv2 # opencv computer vision library, used for image to grayscale, look for an image
import numpy as np # needed since images are just arrays of pixel values, used for array of numbers 
import time

from mss import MSS # library that captures the screen 

# settings 

# mss standard: 0 = all monitors, 1 = main monitor, 2 = second monitor 

MONITOR_NUMBER = 2

# top right of 1920 x 1080 monitor 

REGION_X = 1320
REGION_Y = 0

REGION_WIDTH = 600
REGION_HEIGHT = 400

with MSS() as sct:

    # Get information about the second monitor
    monitor = sct.monitors[MONITOR_NUMBER]

    print("Monitor information:")
    print(monitor)

    # Create our capture region.
    #
    # monitor["left"] and monitor["top"] tell us where
    # monitor 2 begins on the overall Windows desktop.
    region = {
        "left": monitor["left"] + REGION_X,
        "top": monitor["top"] + REGION_Y,
        "width": REGION_WIDTH,
        "height": REGION_HEIGHT
    }

    print("Watching region:")
    print(region)

    while True:

        # Capture the selected region
        screenshot = sct.grab(region)

        # Convert MSS screenshot into a NumPy array
        frame = np.array(screenshot)

        # MSS gives us BGRA.
        # Convert it to BGR so OpenCV can display it normally.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )

        # Show exactly what Python currently sees
        cv2.imshow("Shiny Counter - Capture Preview", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Small delay so we're not capturing unnecessarily fast
        time.sleep(0.05)


cv2.destroyAllWindows()