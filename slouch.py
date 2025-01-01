import os
import sys
import logging
import cv2
import ctypes
import subprocess
import mediapipe as mp
import math
import time
import numpy as np
import winsound
import json
import wx
import threading
import win32event
import win32api
import winerror

# ===========================  RESOURCE PATH HANDLING  ===========================
def resource_path(relative_path):
    """
    Get absolute path to resource, works for development and for PyInstaller.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

# ===========================  SINGLE INSTANCE CHECK  ===========================
def is_another_instance_running():
    """
    Checks if another instance of the application is already running using a global mutex.
    Skips the check if the '--elevated' flag is present.
    """
    if "--elevated" in sys.argv:
        return False  # Elevated instance doesn't perform singleton check

    mutex_name = "Global\\SlouchLockCalibrationWizardMutex"
    try:
        mutex = win32event.CreateMutex(None, False, mutex_name)
        last_error = win32api.GetLastError()
        if last_error == winerror.ERROR_ALREADY_EXISTS:
            return True
        return False
    except Exception as e:
        print(f"Error creating mutex: {e}")
        logging.error(f"Error creating mutex: {e}")
        return False  # Allow the application to run if mutex creation fails

if is_another_instance_running():
    print("Another instance of the application is already running. Exiting.")
    sys.exit(0)

# ===========================  ADMIN CHECK & ELEVATION  ===========================
def is_admin():
    """
    Returns True if the current process has admin privileges (Windows).
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def elevate_to_admin():
    """
    Relaunches the current script with admin privileges using ShellExecute.
    Passes the '--elevated' flag to prevent singleton check in the elevated instance.
    If user denies or it fails, we exit.
    """
    script = resource_path(sys.argv[0])  # Use resource_path for portability
    # Reconstruct the command-line arguments, excluding any existing '--elevated' flags to prevent duplication
    args = [arg for arg in sys.argv[1:] if arg != "--elevated"]
    params = " ".join([f'"{arg}"' for arg in args])  # Preserve script args

    proc_info = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        sys.executable,
        f'"{script}" {params} --elevated',
        None,
        1
    )
    if proc_info <= 32:
        print("User denied admin elevation or error occurred. Exiting.")
        logging.error("User denied admin elevation or ShellExecute failed.")
        sys.exit(1)

# Check admin at the very start
if not is_admin():
    print("Not running as admin. Attempting to relaunch as admin...")
    elevate_to_admin()
    sys.exit(0)

# ===========================  WINDOWS BLOCKINPUT SETUP  ===========================
user32 = ctypes.WinDLL('user32', use_last_error=True)

def block_input(block: bool):
    """
    Block or unblock keyboard/mouse input on Windows using user32.BlockInput.
    Must be run as administrator.
    """
    res = user32.BlockInput(block)
    if res == 0:
        print("[BlockInput] Failed - might lack admin privileges or OS restrictions.")
        logging.error("[BlockInput] Failed to change input blocking state.")
    else:
        state = "Blocked" if block else "Unblocked"
        logging.info(f"Input {state} successfully.")

# ===========================  CONFIG & CONSTANTS  ===========================
# 1. Store calibration data in ProgramData folder under "Slouch" directory
PROGRAMDATA = os.getenv('PROGRAMDATA')
SLOUCH_DIR = os.path.join(PROGRAMDATA, 'Slouch')
if not os.path.exists(SLOUCH_DIR):
    os.makedirs(SLOUCH_DIR)
CALIBRATION_FILE = os.path.join(SLOUCH_DIR, "three_phase_calibration.json")
CONFIG_FILE = os.path.join(SLOUCH_DIR, "config.json")

PHASE_DURATION = 4.0            # seconds for each of the 3 phases
CONSEC_SLOUCH_TO_LOCK = 15
CONSEC_UPRIGHT_TO_UNLOCK = 5
NO_POSE_FRAMES_FOR_SLOUCH = 10  # Define as needed

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
NOSE_IDX = 0
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

# ===========================  HELPER FUNCTIONS  ===========================
def compute_slouch_ratio(frame, landmarks) -> float:
    """
    Compare nose vs. shoulder midpoint, normalized by shoulder width.
    Higher => more forward tilt => slouch.
    """
    h, w, _ = frame.shape
    nose = landmarks[NOSE_IDX]
    lsh = landmarks[LEFT_SHOULDER_IDX]
    rsh = landmarks[RIGHT_SHOULDER_IDX]

    if (nose.visibility < 0.5 or
        lsh.visibility < 0.5 or
        rsh.visibility < 0.5):
        return 0.0

    nose_x, nose_y = nose.x * w, nose.y * h
    lsh_x, lsh_y = lsh.x * w, lsh.y * h
    rsh_x, rsh_y = rsh.x * w, rsh.y * h

    shoulder_mid_y = (lsh_y + rsh_y) / 2.0
    shoulder_width = math.dist((lsh_x, lsh_y), (rsh_x, rsh_y))
    if shoulder_width < 1e-6:
        return 0.0

    vertical_diff = nose_y - shoulder_mid_y
    ratio = vertical_diff / shoulder_width
    return ratio

def compute_ear_shoulder_hip_angle(frame, landmarks) -> float:
    """
    Returns the smaller angle among the left or right side's
      Ear -> Shoulder -> Hip
    Typically near 180 if upright; lower => leaning/hunch.
    """
    h, w, _ = frame.shape

    def angle_for_side(ear_idx, sh_idx, hip_idx):
        ear = landmarks[ear_idx]
        sh  = landmarks[sh_idx]
        hip = landmarks[hip_idx]

        if (ear.visibility < 0.5 or sh.visibility < 0.5 or hip.visibility < 0.5):
            return None

        ex, ey = ear.x * w, ear.y * h
        sx, sy = sh.x * w, sh.y * h
        hx, hy = hip.x * w, hip.y * h

        vSE = (ex - sx, ey - sy)
        vSH = (hx - sx, hy - sy)
        dot   = vSE[0]*vSH[0] + vSE[1]*vSH[1]
        magSE = math.hypot(vSE[0], vSE[1])
        magSH = math.hypot(vSH[0], vSH[1])
        if magSE < 1e-6 or magSH < 1e-6:
            return None

        cos_theta = dot / (magSE * magSH)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    left_angle  = angle_for_side(LEFT_EAR_IDX, LEFT_SHOULDER_IDX, LEFT_HIP_IDX)
    right_angle = angle_for_side(RIGHT_EAR_IDX, RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX)

    if left_angle is None and right_angle is None:
        return None
    if left_angle is None:
        return right_angle
    if right_angle is None:
        return left_angle

    return min(left_angle, right_angle)

def beep():
    """Beep 1200Hz, 200ms."""
    try:
        winsound.Beep(1200, 200)
    except RuntimeError as e:
        print(f"Beep failed: {e}")
        logging.error(f"Beep failed: {e}")

# ===========================  CALIBRATION I/O  ===========================
def load_calibration():
    """
    Loads the calibration profile from CALIBRATION_FILE.
    Expects file: {
      "upright_ratio": float,
      "lookdown_ratio": float,
      "slouch_ratio": float,
      "upright_angle": float,
      "lookdown_angle": float,
      "slouch_angle": float
    }
    """
    if not os.path.exists(CALIBRATION_FILE):
        logging.info("Calibration file not found.")
        return None
    try:
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        needed = ["upright_ratio","lookdown_ratio","slouch_ratio",
                  "upright_angle","lookdown_angle","slouch_angle"]
        if all(k in data for k in needed):
            logging.info("Calibration data loaded successfully.")
            return data
        else:
            logging.warning("Calibration file is missing required keys.")
    except Exception as e:
        print(f"Error loading calibration: {e}")
        logging.error(f"Error loading calibration: {e}")
    return None

def save_calibration(data: dict):
    """
    Saves the calibration profile to CALIBRATION_FILE.
    """
    try:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved calibration to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"Error saving calibration: {e}")
        logging.error(f"Error saving calibration: {e}")

# ===========================  CALIBRATION WIZARD GUI ===========================
class CalibrationWizard(wx.Frame):
    """
    A wizard-style GUI for 3-phase calibration using WxPython.
    Provides instructions, camera feed, and progress bars.
    Accessible to screen readers by managing focus.
    """
    def __init__(self, parent, title, phases, pose_processor):
        super(CalibrationWizard, self).__init__(parent, title=title, size=(1000, 800))
        self.phases = phases  # List of tuples: (label, duration)
        self.pose_processor = pose_processor
        self.current_phase = -1  # Start before the first phase
        self.phase_start_time = None
        self.ratio_vals = []
        self.angle_vals = []
        self.is_calibrating = False
        self.phase_complete = False

        # Initialize OpenCV video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            wx.MessageBox("Error: Could not open webcam.", "Error", wx.OK | wx.ICON_ERROR)
            logging.error("Failed to open webcam.")
            self.Close()
            return
        else:
            logging.info("Webcam opened successfully.")

        # Initialize instance variable for scaled bitmap
        self.scaled_bitmap = None  # Prevent garbage collection

        # Set up the UI
        self.InitUI()

        # Bind the close event
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        # Start the wizard with instructions
        self.Show()

    def InitUI(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)  # Main vertical sizer

        # ------------------- Instruction Panel -------------------
        self.instruction_panel = wx.Panel(panel)
        instruction_sizer = wx.BoxSizer(wx.VERTICAL)

        welcome_text = (
            "Welcome to the Slouch Prevention Calibration Wizard.\n\n"
            "This wizard will guide you through three phases:\n"
            "1. Upright\n"
            "2. Look Down\n"
            "3. Slouch\n\n"
            "Follow the on-screen instructions and ensure your face is clearly visible to the camera during each phase."
        )
        self.instruction_text = wx.StaticText(self.instruction_panel, label=welcome_text)
        self.instruction_text.Wrap(750)

        # Increase font size and set font family for better visibility
        font = self.instruction_text.GetFont()
        font.SetPointSize(16)  # Larger font size
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        font.SetFamily(wx.FONTFAMILY_SWISS)  # Set to a readable font family
        self.instruction_text.SetFont(font)

        instruction_sizer.Add(self.instruction_text, proportion=1, flag=wx.ALIGN_CENTER | wx.ALL, border=15)

        self.start_button = wx.Button(self.instruction_panel, label="Start Calibration (Ctrl+N)")
        self.start_button.Bind(wx.EVT_BUTTON, self.OnStart)
        instruction_sizer.Add(self.start_button, flag=wx.ALIGN_CENTER | wx.TOP, border=10)

        self.instruction_panel.SetSizer(instruction_sizer)
        main_sizer.Add(self.instruction_panel, proportion=1, flag=wx.EXPAND)

        # ------------------- Phase Panel -------------------
        self.phase_panel = wx.Panel(panel)
        phase_sizer = wx.BoxSizer(wx.VERTICAL)

        # Progress Bar
        self.progress = wx.Gauge(self.phase_panel, range=100, size=(800, 25))
        phase_sizer.Add(self.progress, flag=wx.EXPAND | wx.ALL, border=10, proportion=0)

        # Instruction Text
        self.phase_instruction = wx.StaticText(self.phase_panel, label="", style=wx.ALIGN_LEFT)
        phase_sizer.Add(self.phase_instruction, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10, proportion=0)

        # Increase font size and set font family for better visibility
        phase_font = self.phase_instruction.GetFont()
        phase_font.SetPointSize(14)  # Adjusted font size for phase instructions
        phase_font.SetWeight(wx.FONTWEIGHT_BOLD)
        phase_font.SetFamily(wx.FONTFAMILY_SWISS)
        self.phase_instruction.SetFont(phase_font)

        # Camera feed without fixed size
        self.video_ctrl = wx.StaticBitmap(self.phase_panel)
        phase_sizer.Add(self.video_ctrl, flag=wx.EXPAND | wx.ALL, border=10, proportion=1)

        # Next button aligned to bottom right using a horizontal sizer with stretch spacer
        next_button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        next_button_sizer.AddStretchSpacer()
        self.next_button = wx.Button(self.phase_panel, label="Next (Ctrl+N)")
        self.next_button.Disable()
        self.next_button.Bind(wx.EVT_BUTTON, self.OnNext)
        next_button_sizer.Add(self.next_button, flag=wx.ALL, border=10)
        phase_sizer.Add(next_button_sizer, flag=wx.EXPAND)

        self.phase_panel.SetSizer(phase_sizer)
        self.phase_panel.Hide()  # Hide phase panel initially
        main_sizer.Add(self.phase_panel, proportion=3, flag=wx.EXPAND)

        panel.SetSizer(main_sizer)

        # ------------------- Accelerator Keys -------------------
        # Define unique IDs for shortcuts
        self.ID_NEXT_SHORTCUT = wx.Window.NewControlId()
        self.ID_EMERGENCY_UNLOCK = wx.Window.NewControlId()

        # Combined list of accelerator entries
        accel_entries = [
            (wx.ACCEL_CTRL, ord('N'), self.ID_NEXT_SHORTCUT),                   # Ctrl+N for Next
            (wx.ACCEL_CTRL | wx.ACCEL_ALT, ord('U'), self.ID_EMERGENCY_UNLOCK)  # Ctrl+Alt+U for Emergency Unlock
        ]

        # Create a single AcceleratorTable with all entries
        accel_tbl = wx.AcceleratorTable(accel_entries)
        self.SetAcceleratorTable(accel_tbl)

        # Bind the accelerator events to their handlers
        self.Bind(wx.EVT_MENU, self.OnNext, id=self.ID_NEXT_SHORTCUT)
        self.Bind(wx.EVT_MENU, self.EmergencyUnlock, id=self.ID_EMERGENCY_UNLOCK)

        # ------------------- Timer for Camera and Progress -------------------
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)

    def OnStart(self, event):
        logging.info("Start Calibration button clicked.")
        self.instruction_panel.Hide()
        self.phase_panel.Show()

        # **Newly Added Line to Maximize the Window**
        self.Maximize()  # Maximize the window to prevent blank display

        self.Layout()               # Update the layout
        self.phase_panel.Refresh()  # Refresh the phase panel
        self.Refresh()              # Refresh the entire frame
        logging.info("Instruction panel hidden, phase panel shown, and window maximized.")

        # Start the first phase
        self.current_phase = 0
        self.StartPhase()

    def StartPhase(self):
        if self.current_phase >= len(self.phases):
            # Calibration complete
            self.FinishCalibration()
            return

        label, duration = self.phases[self.current_phase]
        self.current_phase_label = label
        self.current_phase_duration = duration
        self.phase_start_time = time.time()
        self.ratio_vals = []
        self.angle_vals = []
        self.phase_complete = False

        # Update instruction with phase-specific guidance
        if self.current_phase == 0:
            instruction = "Phase 1: Start by looking directly at the camera."
        elif self.current_phase == 1:
            instruction = "Phase 2: Now, look down at your keyboard."
        elif self.current_phase == 2:
            instruction = "Phase 3: Finally, slouch slightly to complete the calibration."
        else:
            instruction = f"Phase {self.current_phase +1}: {label}."

        self.phase_instruction.SetLabel(instruction)
        self.phase_instruction.Wrap(750)  # Ensure proper wrapping

        # Set focus to instruction text for screen readers
        self.phase_instruction.SetFocus()
        logging.info(f"Phase {self.current_phase +1} instruction updated.")

        # Reset progress bar
        self.progress.SetValue(0)

        # Start timer if not already
        if not self.is_calibrating:
            self.timer.Start(100)  # Update every 100 ms
            self.is_calibrating = True
            logging.info("Timer started for phase.")

        # Disable Next button until phase is complete
        self.next_button.Disable()
        logging.info("Next button disabled.")

    def OnTimer(self, event):
        ret, frame = self.cap.read()
        if ret:
            try:
                # Convert BGR (OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process pose
                results = self.pose_processor.process(frame_rgb)

                # NOTE: Landmark drawing has been removed as per the requirement

                # No landmark drawing, so frame_rgb remains unchanged

                height, width = frame_rgb.shape[:2]

                # Get the current size of the video control
                video_size = self.video_ctrl.GetSize()
                if video_size.width == 0 or video_size.height == 0:
                    # Default size if not set yet
                    video_size = (800, 600)

                # Create bitmap directly from the buffer
                bitmap = wx.Bitmap.FromBuffer(width, height, frame_rgb)

                # Convert bitmap to image for scaling (compatible with older wxPython versions)
                image = bitmap.ConvertToImage()

                # Scale the image to the current size of the video control
                scaled_image = image.Scale(video_size.width, video_size.height, wx.IMAGE_QUALITY_HIGH)

                # Convert the scaled image back to bitmap and store as an instance variable
                self.scaled_bitmap = wx.Bitmap(scaled_image)  # Store as instance variable to prevent garbage collection

                # Set the bitmap to the video control
                self.video_ctrl.SetBitmap(self.scaled_bitmap)

                # Force the UI to refresh the video control
                self.video_ctrl.Refresh()

                # Update progress bar
                elapsed = time.time() - self.phase_start_time
                progress_percent = min(int((elapsed / self.current_phase_duration) * 100), 100)
                self.progress.SetValue(progress_percent)
                logging.info(f"Progress bar updated to {progress_percent}%.")
                
                # Collect data
                if results.pose_landmarks:
                    ratio = compute_slouch_ratio(frame, results.pose_landmarks.landmark)
                    angle = compute_ear_shoulder_hip_angle(frame, results.pose_landmarks.landmark)
                    if ratio != 0.0:
                        self.ratio_vals.append(ratio)
                    if angle is not None:
                        self.angle_vals.append(angle)
                    logging.info(f"Collected data - Ratio: {ratio}, Angle: {angle}")

                # Check if phase is complete
                if elapsed >= self.current_phase_duration and not self.phase_complete:
                    self.phase_complete = True
                    # Save average values for this phase
                    avg_ratio = np.mean(self.ratio_vals) if self.ratio_vals else 0.0
                    avg_angle = np.mean(self.angle_vals) if self.angle_vals else 180.0

                    logging.info(f"{self.current_phase_label} Phase done. ratio={avg_ratio:.3f}, angle={avg_angle:.1f}\n")

                    # Store calibration data
                    if self.current_phase == 0:
                        self.upright_ratio = avg_ratio
                        self.upright_angle = avg_angle
                    elif self.current_phase == 1:
                        self.lookdown_ratio = avg_ratio
                        self.lookdown_angle = avg_angle
                    elif self.current_phase == 2:
                        self.slouch_ratio = avg_ratio
                        self.slouch_angle = avg_angle

                    # Stop timer temporarily
                    self.timer.Stop()
                    self.is_calibrating = False
                    logging.info("Timer stopped after phase completion.")

                    # Update instruction for the next phase or completion
                    if self.current_phase < len(self.phases) - 1:
                        self.phase_instruction.SetLabel("Press Next (Ctrl+N) to begin the next calibration phase.")
                        logging.info("Instruction updated for next phase.")
                    else:
                        self.phase_instruction.SetLabel("Calibration done! Press Finish (Ctrl+N) to start using the application.")
                        logging.info("Instruction updated for calibration completion.")

                    self.phase_instruction.Wrap(750)  # Ensure proper wrapping

                    # Set focus to instruction text for screen readers
                    self.phase_instruction.SetFocus()

                    # Enable Next button (label changes to "Finish (Ctrl+N)" on last phase)
                    if self.current_phase == len(self.phases) - 1:
                        self.next_button.SetLabel("Finish (Ctrl+N)")
                    else:
                        self.next_button.SetLabel("Next (Ctrl+N)")
                    self.next_button.Enable()
                    logging.info("Next button enabled.")

            except Exception as e:
                error_msg = f"Error during OnTimer processing: {e}"
                print(error_msg)
                logging.error(error_msg)
                self.timer.Stop()
                self.is_calibrating = False

                # Display detailed error message to the user
                wx.MessageBox(f"An error occurred while processing the video feed:\n{e}", "Error", wx.OK | wx.ICON_ERROR)

    def OnNext(self, event):
        logging.info("Next button clicked.")
        # Disable Next button
        self.next_button.Disable()

        # Proceed to next phase
        self.current_phase += 1
        if self.current_phase < len(self.phases):
            self.StartPhase()
        else:
            self.FinishCalibration()

    def FinishCalibration(self):
        logging.info("Finishing calibration and saving data.")
        # Save calibration data
        profile = {
            "upright_ratio": getattr(self, 'upright_ratio', 0.0),
            "lookdown_ratio": getattr(self, 'lookdown_ratio', 0.0),
            "slouch_ratio": getattr(self, 'slouch_ratio', 0.0),
            "upright_angle": getattr(self, 'upright_angle', 180.0),
            "lookdown_angle": getattr(self, 'lookdown_angle', 180.0),
            "slouch_angle": getattr(self, 'slouch_angle', 180.0)
        }
        save_calibration(profile)

        # Log calibration completion
        logging.info("Calibration complete!")

        # Stop timer and release camera
        self.timer.Stop()
        self.cap.release()
        logging.info("Timer stopped and webcam released.")

        # Update instruction to indicate completion
        self.phase_instruction.SetLabel("Calibration complete! The system will now start monitoring your posture.")
        self.phase_instruction.Wrap(750)  # Ensure proper wrapping

        # Set focus to instruction text for screen readers
        self.phase_instruction.SetFocus()
        logging.info("Calibration completion instruction updated.")

        # Show completion message before closing
        wx.MessageBox("Calibration complete! The system will now start monitoring your posture.", "Calibration Complete", wx.OK | wx.ICON_INFORMATION)

        # Close the wizard
        self.Close()

    def OnClose(self, event):
        logging.info("Application closing.")
        # Ensure resources are released
        if self.is_calibrating:
            self.timer.Stop()
            logging.info("Timer stopped during application close.")
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Webcam released during application close.")
        self.Destroy()

    # Emergency Unlock Method
    def EmergencyUnlock(self, event):
        logging.warning("Emergency Unlock triggered.")
        block_input(False)
        print("==== EMERGENCY UNLOCK TRIGGERED ====")
        wx.MessageBox("Emergency unlock activated. Input has been unblocked.", "Emergency Unlock", wx.OK | wx.ICON_WARNING)
        logging.info("Input unblocked via Emergency Unlock.")
        self.Close()

# ===========================  MAIN SCRIPT  ===========================
def main():
    # Initialize logging
    if not os.path.exists(SLOUCH_DIR):
        os.makedirs(SLOUCH_DIR)
    logging.basicConfig(
        filename=os.path.join(SLOUCH_DIR, 'slouch_log.log'),
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logging.info("Application started.")

    print("=== Slouch Lock with 3-Phase Calibration (Upright, LookDown, Slouch) ===\n")

    # Attempt to load existing calibration
    profile = load_calibration()

    if profile:
        print("Loaded existing calibration profile:\n", profile)
        logging.info("Loaded existing calibration profile.")
        print("Skipping calibration phases.\n")
    else:
        # Initialize Wx App for Calibration Wizard
        app = wx.App(False)
        phases = [
            ("Upright", PHASE_DURATION),
            ("LookDown", PHASE_DURATION),
            ("Slouch", PHASE_DURATION)
        ]
        calibration_wizard = CalibrationWizard(None, "Calibration Wizard", phases, pose)
        app.MainLoop()

        # After calibration, load the profile
        profile = load_calibration()
        if not profile:
            print("Calibration failed. Exiting.")
            logging.error("Calibration failed.")
            return

    # Prepare final data
    # Define "ratio" zones: 
    #   zone1 < mid(upright, lookdown) => upright
    #   zone1 < mid(lookdown, slouch)  => look down
    #   else => slouch
    ratio_up_down_cut = (profile["upright_ratio"] + profile["lookdown_ratio"]) / 2.0
    ratio_down_slouch_cut = (profile["lookdown_ratio"] + profile["slouch_ratio"]) / 2.0

    # For angle, typically upright_angle > lookdown_angle > slouch_angle
    # Define:
    #  if angle > mid(lookdown_angle, upright_angle) => upright
    #  if angle > mid(slouch_angle, lookdown_angle)  => look down
    #  else => slouch
    angle_up_down_cut = (profile["upright_angle"] + profile["lookdown_angle"]) / 2.0
    angle_down_slouch_cut = (profile["lookdown_angle"] + profile["slouch_angle"]) / 2.0

    print("Computed ratio cutoffs:")
    print(f"  upright/lookdown => {ratio_up_down_cut:.3f}")
    print(f"  lookdown/slouch  => {ratio_down_slouch_cut:.3f}")

    print("Computed angle cutoffs:")
    print(f"  upright/lookdown => {angle_up_down_cut:.1f}")
    print(f"  lookdown/slouch  => {angle_down_slouch_cut:.1f}\n")

    logging.info(f"Computed ratio cutoffs: upright/lookdown={ratio_up_down_cut:.3f}, lookdown/slouch={ratio_down_slouch_cut:.3f}")
    logging.info(f"Computed angle cutoffs: upright/lookdown={angle_up_down_cut:.1f}, lookdown/slouch={angle_down_slouch_cut:.1f}")

    # Monitoring
    cap_monitor = cv2.VideoCapture(0)
    if not cap_monitor.isOpened():
        print("Error: Could not open webcam for monitoring.")
        logging.error("Could not open webcam for monitoring.")
        return

    # Optimize frame resolution for performance
    monitor_frame_width = 320
    monitor_frame_height = 240
    cap_monitor.set(cv2.CAP_PROP_FRAME_WIDTH, monitor_frame_width)
    cap_monitor.set(cv2.CAP_PROP_FRAME_HEIGHT, monitor_frame_height)

    # Prepare for monitoring
    consecutive_slouch = 0
    consecutive_upright = 0
    locked = False
    no_pose_count = 0

    last_beep_time = 0.0
    BEEP_INTERVAL = 0.01  # seconds between beeps to prevent rapid beeping

    print(f"Monitoring posture. If {CONSEC_SLOUCH_TO_LOCK} consecutive 'slouch' frames => lock input.")
    print(f"If no pose for {NO_POSE_FRAMES_FOR_SLOUCH} frames => slouch.\n")
    logging.info("Monitoring started.")

    # Define a function for monitoring in a separate thread
    def monitoring_loop(profile_data):
        nonlocal consecutive_slouch, consecutive_upright, locked, no_pose_count
        nonlocal last_beep_time

        try:
            while True:
                ret, frame = cap_monitor.read()
                if not ret:
                    print("Camera read failed. Exiting.")
                    logging.error("Camera read failed during monitoring.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                # Interpret ratio_category & angle_category each as "upright", "lookdown", or "slouch"
                ratio_cat = "upright"
                angle_cat = "upright"

                # Default is no slouch
                is_slouch = False

                if results.pose_landmarks:
                    no_pose_count = 0
                    ratio = compute_slouch_ratio(frame, results.pose_landmarks.landmark)
                    angle = compute_ear_shoulder_hip_angle(frame, results.pose_landmarks.landmark)

                    # Categorize ratio
                    if ratio < profile_data["upright_ratio"] + (profile_data["lookdown_ratio"] - profile_data["upright_ratio"]) / 2.0:
                        ratio_cat = "upright"
                    elif ratio < profile_data["lookdown_ratio"] + (profile_data["slouch_ratio"] - profile_data["lookdown_ratio"]) / 2.0:
                        ratio_cat = "lookdown"
                    else:
                        ratio_cat = "slouch"

                    # Categorize angle
                    if angle is None:
                        angle_cat = "upright"  # Fallback
                    else:
                        # Angles are reversed: bigger angle => more upright
                        if angle > profile_data["upright_angle"] + (profile_data["lookdown_angle"] - profile_data["upright_angle"]) / 2.0:
                            angle_cat = "upright"
                        elif angle > profile_data["lookdown_angle"] + (profile_data["slouch_angle"] - profile_data["lookdown_angle"]) / 2.0:
                            angle_cat = "lookdown"
                        else:
                            angle_cat = "slouch"

                    # Decide final slouch if either ratio_cat == "slouch" OR angle_cat == "slouch"
                    is_slouch = (ratio_cat == "slouch" or angle_cat == "slouch")
                else:
                    # No pose => increment no_pose_count
                    no_pose_count += 1
                    if no_pose_count >= NO_POSE_FRAMES_FOR_SLOUCH:
                        is_slouch = True

                # Normal logic
                if not locked:
                    if is_slouch:
                        consecutive_slouch += 1
                        logging.info(f"Slouch detected: {consecutive_slouch} consecutive slouches.")
                    else:
                        if consecutive_slouch > 0:
                            logging.info(f"Slouch streak broken. Consecutive slouches reset from {consecutive_slouch} to 0.")
                        consecutive_slouch = 0
                    consecutive_upright = 0

                    if consecutive_slouch >= CONSEC_SLOUCH_TO_LOCK:
                        print("==== SLOUCH DETECTED => BLOCKING INPUT ====")
                        logging.warning("Slouch detected. Blocking input.")
                        block_input(True)
                        locked = True
                        last_beep_time = time.time()
                        consecutive_slouch = 0
                else:
                    # Locked => beep & check upright
                    now = time.time()
                    if now - last_beep_time > BEEP_INTERVAL:
                        beep()
                        last_beep_time = now

                    if not is_slouch:
                        consecutive_upright += 1
                        logging.info(f"Upright detected: {consecutive_upright} consecutive uprights.")
                    else:
                        if consecutive_upright > 0:
                            logging.info(f"Upright streak broken. Consecutive uprights reset from {consecutive_upright} to 0.")
                        consecutive_upright = 0

                    if consecutive_upright >= CONSEC_UPRIGHT_TO_UNLOCK:
                        print("==== USER SAT UPRIGHT => UNBLOCKING INPUT ====")
                        logging.info("User returned to upright posture. Unblocking input.")
                        block_input(False)
                        locked = False
                        consecutive_upright = 0

                # To prevent high CPU usage, sleep for a short duration
                time.sleep(0.01)

        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            logging.error(f"Error in monitoring loop: {e}")
        finally:
            block_input(False)
            cap_monitor.release()
            logging.info("Monitoring loop ended.")

    # Start the monitoring loop in a separate thread
    monitoring_thread = threading.Thread(target=monitoring_loop, args=(profile,), daemon=True)
    monitoring_thread.start()

    # Keep the main thread alive to allow monitoring to continue
    try:
        while monitoring_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
        logging.info("Interrupted by user.")
    finally:
        block_input(False)
        cap_monitor.release()
        logging.info("Slouch Lock script ended.")

if __name__ == "__main__":
    main()
