import queue
import threading
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
from mss import MSS


# ============================================================
# SCREEN-DETECTION SETTINGS
# ============================================================

# MSS monitor numbers: 0 = all monitors, 1 = main monitor,
# 2 = second monitor.
MONITOR_NUMBER = 2

# Top-right area of a 1920 x 1080 monitor.
REGION_X = 1320
REGION_Y = 0
REGION_WIDTH = 600
REGION_HEIGHT = 400

DETECTION_THRESHOLD = 0.95
RELEASE_THRESHOLD = 0.70

# Build paths from this script's folder so the program still works when it is
# started from a shortcut or from another working directory.
APP_FOLDER = Path(__file__).resolve().parent
TEMPLATE_PATH = APP_FOLDER / "reset_templates" / "reset_template.png"
COUNT_FILE = APP_FOLDER / "count.txt"


# ============================================================
# SAVING AND LOADING THE COUNTER
# ============================================================

def load_count():
    """Load the previous reset count, or start at zero if it is unavailable."""
    try:
        return int(COUNT_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return 0


def save_count(count):
    """Save the current reset count to count.txt."""
    COUNT_FILE.write_text(str(count), encoding="utf-8")


class ShinyCounterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.count = load_count()

        # The lock protects the counter when the GUI and detection thread try
        # to change it at the same time.
        self.count_lock = threading.Lock()

        # The worker puts updates here. The GUI checks the queue with after(),
        # which keeps all widget updates safely on CustomTkinter's main thread.
        self.ui_updates = queue.Queue()
        self.stop_event = threading.Event()

        self.setup_gui()

        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.after(50, self.process_ui_updates)

        # Run screen capture and template matching away from the GUI thread so
        # the window remains responsive.
        self.detection_thread = threading.Thread(
            target=self.detection_loop,
            name="screen-detection",
            daemon=True,
        )
        self.detection_thread.start()

    # ========================================================
    # GUI SETUP
    # ========================================================

    def setup_gui(self):
        """Create the small, light CustomTkinter interface."""
        self.title("Shiny Hunt Counter")
        self.geometry("350x340")
        self.resizable(False, False)
        self.configure(fg_color="#F7F8FA")

        self.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            self,
            text="Shiny Hunt Counter",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#20242A",
        )
        title_label.grid(row=0, column=0, padx=24, pady=(20, 12))

        counter_card = ctk.CTkFrame(
            self,
            width=302,
            height=135,
            corner_radius=18,
            fg_color="#FFFFFF",
            border_width=1,
            border_color="#E5E7EB",
        )
        counter_card.grid(row=1, column=0, padx=24, sticky="ew")
        counter_card.grid_propagate(False)
        counter_card.grid_columnconfigure(0, weight=1)

        self.count_label = ctk.CTkLabel(
            counter_card,
            text=str(self.count),
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color="#111827",
        )
        self.count_label.grid(row=0, column=0, pady=(12, 0))

        resets_label = ctk.CTkLabel(
            counter_card,
            text="Resets",
            font=ctk.CTkFont(size=14),
            text_color="#6B7280",
        )
        resets_label.grid(row=1, column=0, pady=(0, 5))

        self.match_label = ctk.CTkLabel(
            counter_card,
            text="Match: 0.00%",
            font=ctk.CTkFont(size=13),
            text_color="#4B5563",
        )
        self.match_label.grid(row=2, column=0)

        self.status_label = ctk.CTkLabel(
            self,
            text="Watching...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2563EB",
        )
        self.status_label.grid(row=2, column=0, pady=(10, 8))

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.grid(row=3, column=0)

        minus_button = ctk.CTkButton(
            button_row,
            text="-1",
            width=125,
            height=36,
            corner_radius=10,
            fg_color="#E5E7EB",
            hover_color="#D1D5DB",
            text_color="#1F2937",
            command=lambda: self.change_count(-1),
        )
        minus_button.grid(row=0, column=0, padx=(0, 8))

        plus_button = ctk.CTkButton(
            button_row,
            text="+1",
            width=125,
            height=36,
            corner_radius=10,
            command=lambda: self.change_count(1),
        )
        plus_button.grid(row=0, column=1, padx=(8, 0))

        reset_button = ctk.CTkButton(
            self,
            text="Reset Counter",
            width=266,
            height=36,
            corner_radius=10,
            fg_color="#FFFFFF",
            hover_color="#F3F4F6",
            border_width=1,
            border_color="#D1D5DB",
            text_color="#374151",
            command=self.reset_count,
        )
        reset_button.grid(row=4, column=0, pady=(10, 16))

    # ========================================================
    # COUNTER BUTTONS
    # ========================================================

    def change_count(self, amount):
        """Increment or decrement the count and save it immediately."""
        with self.count_lock:
            self.count = max(0, self.count + amount)
            save_count(self.count)
            new_count = self.count

        self.count_label.configure(text=str(new_count))

    def reset_count(self):
        """Set the counter back to zero and save the change."""
        with self.count_lock:
            self.count = 0
            save_count(self.count)

        self.count_label.configure(text="0")

    # ========================================================
    # GUI UPDATES FROM THE BACKGROUND THREAD
    # ========================================================

    def process_ui_updates(self):
        """Apply queued detector updates on the CustomTkinter main thread."""
        try:
            while True:
                update_type, value = self.ui_updates.get_nowait()

                if update_type == "count":
                    # Read the latest shared value instead of trusting an older
                    # queued value (the user may have clicked a button since).
                    with self.count_lock:
                        current_count = self.count
                    self.count_label.configure(text=str(current_count))
                elif update_type == "match":
                    self.match_label.configure(text=f"Match: {value:.2f}%")
                elif update_type == "status":
                    color = "#16A34A" if value == "Reset Detected" else "#2563EB"
                    self.status_label.configure(text=value, text_color=color)
        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.after(50, self.process_ui_updates)

    # ========================================================
    # BACKGROUND SCREEN CAPTURE AND TEMPLATE DETECTION
    # ========================================================

    def detection_loop(self):
        """Continuously capture the selected region and look for the template."""
        template = cv2.imread(str(TEMPLATE_PATH))

        if template is None:
            message = f"Template not found: {TEMPLATE_PATH}"
            print(f"ERROR: {message}")
            self.ui_updates.put(("status", "Template Not Found"))
            return

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        already_detected = False

        try:
            with MSS() as sct:
                monitor = sct.monitors[MONITOR_NUMBER]
                region = {
                    "left": monitor["left"] + REGION_X,
                    "top": monitor["top"] + REGION_Y,
                    "width": REGION_WIDTH,
                    "height": REGION_HEIGHT,
                }

                print(f"Watching region: {region}")

                while not self.stop_event.is_set():
                    # Capture the selected screen region invisibly. MSS returns
                    # BGRA pixels, so convert them to OpenCV's BGR format.
                    screenshot = sct.grab(region)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Compare the current frame with the saved reset template.
                    result = cv2.matchTemplate(
                        gray_frame,
                        template_gray,
                        cv2.TM_CCOEFF_NORMED,
                    )
                    _, max_score, _, _ = cv2.minMaxLoc(result)
                    match_percent = max_score * 100
                    self.ui_updates.put(("match", match_percent))

                    # Count a reset once when the detection threshold is reached.
                    if max_score >= DETECTION_THRESHOLD:
                        if not already_detected:
                            with self.count_lock:
                                self.count += 1
                                save_count(self.count)
                                new_count = self.count

                            self.ui_updates.put(("count", new_count))
                            self.ui_updates.put(("status", "Reset Detected"))

                            print("RESET DETECTED!")
                            print(f"Count: {new_count}")
                            print(f"Match: {match_percent:.2f}%")

                            # Do not count this same reset screen again.
                            already_detected = True

                    # Re-arm only after the reset image has clearly disappeared.
                    elif max_score <= RELEASE_THRESHOLD:
                        if already_detected:
                            self.ui_updates.put(("status", "Watching..."))
                        already_detected = False

                    # Approximately 20 checks per second. Event.wait() also lets
                    # the thread stop promptly when the window is closed.
                    self.stop_event.wait(0.05)

        except (IndexError, OSError, cv2.error) as error:
            print(f"Detection error: {error}")
            self.ui_updates.put(("status", "Detection Error"))

    def close_app(self):
        """Ask the worker thread to stop, then close the GUI."""
        self.stop_event.set()
        print(f"Final count: {self.count}")
        print("Shiny counter closed.")
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    app = ShinyCounterApp()
    app.mainloop()
