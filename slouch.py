import sys
import os
import ctypes
import subprocess
import cv2
import mediapipe as mp
import math
import time
import numpy as np
import winsound
import json

#
# ===========================  ADMIN CHECK & ELEVATION  ===========================
#
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
    If user denies or it fails, we exit.
    """
    script = os.path.abspath(sys.argv[0])
    params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])  # preserve script args

    proc_info = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        sys.executable,
        f'"{script}" {params}',
        None,
        1
    )
    if proc_info <= 32:
        print("User denied admin elevation or error occurred. Exiting.")
        sys.exit(1)

# Check admin at the very start
if not is_admin():
    print("Not running as admin. Attempting to relaunch as admin...")
    elevate_to_admin()
    sys.exit(0)

#
# ===========================  WINDOWS BLOCKINPUT SETUP  ===========================
#
user32 = ctypes.WinDLL('user32', use_last_error=True)
def block_input(block: bool):
    """
    Block or unblock keyboard/mouse input on Windows using user32.BlockInput.
    Must be run as administrator.
    """
    res = user32.BlockInput(block)
    if res == 0:
        print("[BlockInput] Failed - might lack admin privileges or OS restrictions.")


#
# ===========================  CONFIG & CONSTANTS  ===========================
#
CALIBRATION_FILE = "three_phase_calibration.json"

PHASE_DURATION = 4.0            # seconds for each of the 3 phases
CONSEC_SLOUCH_TO_LOCK = 15
CONSEC_UPRIGHT_TO_UNLOCK = 5

BEEP_INTERVAL = 0.3
TTS_INTERVAL = 2.0
NO_POSE_FRAMES_FOR_SLOUCH = 15

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

#
# ===========================  HELPER FUNCTIONS  ===========================
#
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
    winsound.Beep(1200, 200)

def speak_slouch_message():
    """Use PowerShell TTS (async)."""
    message = "You are slouching. Please sit up now!"
    powershell_command = [
        "powershell",
        "-Command",
        f"""
        Add-Type -AssemblyName System.Speech;
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
        $speak.Speak("{message}");
        """
    ]
    subprocess.Popen(powershell_command, shell=True)

#
# ===========================  CALIBRATION I/O  ===========================
#
def load_calibration():
    """
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
        return None
    try:
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        needed = ["upright_ratio","lookdown_ratio","slouch_ratio",
                  "upright_angle","lookdown_angle","slouch_angle"]
        if all(k in data for k in needed):
            return data
    except:
        pass
    return None

def save_calibration(data: dict):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)

#
# ===========================  3-PHASE CALIBRATION  ===========================
#
def calibration_phase(cap, label: str, duration: float):
    """
    Captures ratio & angle for 'duration' seconds.
    label can be "Upright", "LookDown", or "Slouch".
    Returns (avg_ratio, avg_angle).
    """
    print(f"** {label} Phase: Please {label} for ~{duration} seconds **")
    start = time.time()
    ratio_vals = []
    angle_vals = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed in calibration.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        elapsed = time.time() - start
        if results.pose_landmarks:
            ratio = compute_slouch_ratio(frame, results.pose_landmarks.landmark)
            angle = compute_ear_shoulder_hip_angle(frame, results.pose_landmarks.landmark)
            if ratio != 0.0:
                ratio_vals.append(ratio)
            if angle is not None:
                angle_vals.append(angle)

        if elapsed > duration:
            break

        cv2.waitKey(1)

    if len(ratio_vals) == 0:
        mean_ratio = 0.0
    else:
        mean_ratio = np.mean(ratio_vals)

    if len(angle_vals) == 0:
        mean_angle = 180.0
    else:
        mean_angle = np.mean(angle_vals)

    print(f"{label} Phase done. ratio={mean_ratio:.3f}, angle={mean_angle:.1f}\n")
    return (mean_ratio, mean_angle)


#
# ===========================  MAIN SCRIPT  ===========================
#
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("=== Slouch Lock with 3-Phase Calibration (Upright, LookDown, Slouch) ===\n")

    # Attempt to load existing calibration
    profile = load_calibration()

    if profile:
        print("Loaded existing calibration profile:\n", profile)
        print("Skipping calibration phases.\n")
    else:
        print("No existing calibration file. Starting 3-phase calibration...\n")

        # Phase 1: Upright
        upright_ratio, upright_angle = calibration_phase(cap, "Upright", PHASE_DURATION)
        # Phase 2: LookDown
        lookdown_ratio, lookdown_angle = calibration_phase(cap, "LookDown", PHASE_DURATION)
        # Phase 3: Slouch
        slouch_ratio, slouch_angle = calibration_phase(cap, "Slouch", PHASE_DURATION)

        profile = {
            "upright_ratio": upright_ratio,
            "lookdown_ratio": lookdown_ratio,
            "slouch_ratio": slouch_ratio,
            "upright_angle": upright_angle,
            "lookdown_angle": lookdown_angle,
            "slouch_angle": slouch_angle
        }
        save_calibration(profile)
        print(f"Saved calibration to {CALIBRATION_FILE}\n")

    # Prepare final data
    # We'll define "ratio" zones: 
    #   zone1 < mid(upright, lookdown) => upright
    #   zone1 < mid(lookdown, slouch)  => look down
    #   else => slouch
    ratio_up_down_cut = (profile["upright_ratio"] + profile["lookdown_ratio"]) / 2.0
    ratio_down_slouch_cut = (profile["lookdown_ratio"] + profile["slouch_ratio"]) / 2.0

    # For angle, typically upright_angle > lookdown_angle > slouch_angle
    # We'll define:
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

    # Monitoring
    consecutive_slouch = 0
    consecutive_upright = 0
    locked = False
    no_pose_count = 0

    last_beep_time = 0.0
    last_tts_time = 0.0

    print(f"Monitoring posture. If {CONSEC_SLOUCH_TO_LOCK} consecutive 'slouch' frames => lock input.")
    print(f"If no pose for {NO_POSE_FRAMES_FOR_SLOUCH} frames => slouch.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed. Exiting.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # We'll interpret ratio_category & angle_category each as "upright", "lookdown", or "slouch"
        ratio_cat = "upright"
        angle_cat = "upright"

        # default is no slouch
        is_slouch = False

        if results.pose_landmarks:
            no_pose_count = 0
            ratio = compute_slouch_ratio(frame, results.pose_landmarks.landmark)
            angle = compute_ear_shoulder_hip_angle(frame, results.pose_landmarks.landmark)

            # categorize ratio
            if ratio < ratio_up_down_cut:
                ratio_cat = "upright"
            elif ratio < ratio_down_slouch_cut:
                ratio_cat = "lookdown"
            else:
                ratio_cat = "slouch"

            # categorize angle
            if angle is None:
                angle_cat = "upright"  # fallback
            else:
                # angles are reversed: bigger angle => more upright
                if angle > angle_up_down_cut:
                    angle_cat = "upright"
                elif angle > angle_down_slouch_cut:
                    angle_cat = "lookdown"
                else:
                    angle_cat = "slouch"

            # Decide final slouch if either ratio_cat == "slouch" OR angle_cat == "slouch"
            is_slouch = (ratio_cat == "slouch" or angle_cat == "slouch")
        else:
            # no pose => increment no_pose_count
            no_pose_count += 1
            if no_pose_count >= NO_POSE_FRAMES_FOR_SLOUCH:
                is_slouch = True

        # normal logic
        if not locked:
            if is_slouch:
                consecutive_slouch += 1
            else:
                consecutive_slouch = 0
            consecutive_upright = 0

            if consecutive_slouch >= CONSEC_SLOUCH_TO_LOCK:
                print("==== SLOUCH DETECTED => BLOCKING INPUT ====")
                block_input(True)
                locked = True
                last_beep_time = time.time()
                last_tts_time = time.time()
                consecutive_slouch = 0
        else:
            # locked => beep & TTS & check upright
            now = time.time()
            if now - last_beep_time > BEEP_INTERVAL:
                beep()
                last_beep_time = now

            if now - last_tts_time > TTS_INTERVAL:
                speak_slouch_message()
                last_tts_time = now

            if not is_slouch:
                consecutive_upright += 1
            else:
                consecutive_upright = 0

            if consecutive_upright >= CONSEC_UPRIGHT_TO_UNLOCK:
                print("==== USER SAT UPRIGHT => UNBLOCKING INPUT ====")
                block_input(False)
                locked = False
                consecutive_upright = 0

        cv2.waitKey(1)

    block_input(False)
    cap.release()
    print("Slouch Lock script ended.")

if __name__ == "__main__":
    main()
