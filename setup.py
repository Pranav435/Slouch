import sys
import os
from cx_Freeze import setup, Executable
import mediapipe

# Function to collect MediaPipe .tflite files
def collect_mediapipe_tflite_files():
    mediapipe_package_dir = os.path.dirname(mediapipe.__file__)
    mediapipe_modules_dir = os.path.join(mediapipe_package_dir, 'modules')

    tflite_files = []
    for root, dirs, files in os.walk(mediapipe_modules_dir):
        for file in files:
            if file.endswith('.tflite'):
                full_path = os.path.join(root, file)
                # Compute the relative path to mediapipe_package_dir
                relative_path = os.path.relpath(full_path, mediapipe_package_dir)
                # Destination path within the build directory
                dest_path = os.path.join('mediapipe', 'modules', os.path.dirname(relative_path))
                tflite_files.append((full_path, dest_path))
    return tflite_files

# Collect additional data files
additional_datas = collect_mediapipe_tflite_files()

# Define build options
build_exe_options = {
    "packages": [
        "mediapipe",
        "cv2",
        "wx",
        "win32event",
        "win32api",
        "winerror",
        "numpy",
        "ctypes",
        "json",
        "logging",
        "winsound",
        "threading",
        "subprocess",
        "math",
        "time",
        "sys",
        "os",
    ],
    "includes": [],  # Removed "mediapipe.*"
    "include_files": additional_datas,
    "excludes": ["cv2.gapi.wip"],  # Exclude the problematic module
    "optimize": 2,
    "build_exe": "build/exe",  # Ensure this is different from build_base
}

# Base setup
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use "Win32GUI" for GUI applications to hide the console
    # If you need the console for debugging, set base = None or comment out the line

setup(
    name="slouch",
    version="1.0",
    description="Slouch Prevention Application",
    options={"build_exe": build_exe_options},
    executables=[Executable("slouch.py", base=base)],
)
