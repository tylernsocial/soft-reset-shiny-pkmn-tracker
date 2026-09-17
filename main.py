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

TEMPLATE_PATH = "./reset_templates/reset_template.png"


# Load our reference reset image
template = cv2.imread(TEMPLATE_PATH)

# Make sure OpenCV actually found the image
if template is None:
    print("ERROR: Could not load reset template.")
    print(f"Checked: {TEMPLATE_PATH}")
    exit()

# Convert template to grayscale
template_gray = cv2.cvtColor(
    template,
    cv2.COLOR_BGR2GRAY
)

with MSS() as sct:

    monitor = sct.monitors[MONITOR_NUMBER]

    region = {
        "left": monitor["left"] + REGION_X,
        "top": monitor["top"] + REGION_Y,
        "width": REGION_WIDTH,
        "height": REGION_HEIGHT
    }

    while True:

        # Capture the selected part of monitor 2
        screenshot = sct.grab(region)

        # Convert the screenshot into a NumPy array
        frame = np.array(screenshot)

        # Convert BGRA -> BGR
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )

        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # --------------------------
        # TEMPLATE MATCHING
        # --------------------------

        result = cv2.matchTemplate(
            gray_frame,
            template_gray,
            cv2.TM_CCOEFF_NORMED
        )

        # Get the best match score
        _, max_score, _, max_location = cv2.minMaxLoc(result)

        # Convert 0.95 -> 95%
        match_percent = max_score * 100

        print(f"Match: {match_percent:.2f}%")


        # --------------------------
        # DISPLAY
        # --------------------------

        cv2.imshow(
            "Shiny Counter - Capture Preview",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Press S to overwrite the saved reset template
        if key == ord("s"):
            cv2.imwrite(TEMPLATE_PATH, frame)
            print("Reset template saved!")

        # Press Q to quit
        if key == ord("q"):
            break

        time.sleep(0.05)


cv2.destroyAllWindows()