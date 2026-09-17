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

DETECTION_THRESHOLD = 0.95

RELEASE_THRESHOLD = 0.70
# Load our reference reset image
template = cv2.imread(TEMPLATE_PATH)


COUNT_FILE = "count.txt" 

# Load the previous count from count.txt
def load_count():
    try:
        with open(COUNT_FILE, "r") as file:
            return int(file.read())
    except (FileNotFoundError, ValueError):
        return 0


# Save the current count to count.txt
def save_count(count):
    with open(COUNT_FILE, "w") as file:
        file.write(str(count))


# Starting counter value
count = load_count()

# Used so one reset screen only counts once
already_detected = False

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

    # Get information about monitor 2
    monitor = sct.monitors[MONITOR_NUMBER]

    # Define the specific region we want to watch
    region = {
        "left": monitor["left"] + REGION_X,
        "top": monitor["top"] + REGION_Y,
        "width": REGION_WIDTH,
        "height": REGION_HEIGHT
    }

    print(f"Watching region: {region}")


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Capture screen
        # ----------------------------------------------------

        screenshot = sct.grab(region)

        # Convert the MSS screenshot into a NumPy array
        frame = np.array(screenshot)

        # MSS gives us BGRA.
        # Convert it to normal OpenCV BGR.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )


        # ----------------------------------------------------
        # Convert current frame to grayscale
        # ----------------------------------------------------

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Compare current screen against reset template
        # ----------------------------------------------------

        result = cv2.matchTemplate(
            gray_frame,
            template_gray,
            cv2.TM_CCOEFF_NORMED
        )

        # Find the strongest match
        _, max_score, _, _ = cv2.minMaxLoc(result)

        match_percent = max_score * 100


        # ----------------------------------------------------
        # RESET DETECTION
        # ----------------------------------------------------

        # Reset frame has appeared
        if max_score >= DETECTION_THRESHOLD:

            # Only count it if this is a NEW detection
            if not already_detected:

                count += 1

                print()
                print("============================")
                print("RESET DETECTED!")
                print(f"Count: {count}")
                print(f"Match: {match_percent:.2f}%")
                print("============================")

                # Save the new counter value
                save_count(count)

                # Prevent this same reset screen
                # from being counted again
                already_detected = True


        # ----------------------------------------------------
        # RE-ARM DETECTOR
        # ----------------------------------------------------

        # Don't re-arm until the reset image has clearly
        # disappeared from the screen.
        elif max_score <= RELEASE_THRESHOLD:

            already_detected = False


        # ----------------------------------------------------
        # DISPLAY INFORMATION ON PREVIEW
        # ----------------------------------------------------

        # Show current count
        cv2.putText(
            frame,
            f"Resets: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Show current template match %
        cv2.putText(
            frame,
            f"Match: {match_percent:.2f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Show preview
        # ----------------------------------------------------

        cv2.imshow(
            "Shiny Counter - Capture Preview",
            frame
        )


        # ----------------------------------------------------
        # Keyboard input
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Press Q to quit
        if key == ord("q"):
            break


        # About 20 checks per second
        time.sleep(0.05)


# ============================================================
# CLEANUP
# ============================================================

cv2.destroyAllWindows()

print()
print(f"Final count: {count}")
print("Shiny counter closed.")