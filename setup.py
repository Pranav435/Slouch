import sys
import os
from cx_Freeze import setup, Executable
import mediapipe

# ===========================  MEDIA PIPE TFLITE FILES COLLECTION  ===========================
def collect_mediapipe_tflite_files():
    """
    Collect all .tflite files from the mediapipe.modules directory.
    Returns a list of tuples with (source_file, destination_folder).
    """
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

# ===========================  COLLECT ADDITIONAL DATA FILES  ===========================
additional_datas = collect_mediapipe_tflite_files()

# ===========================  BUILD EXE OPTIONS  ===========================
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
        "psutil",  # Added psutil for process management
    ],
    "includes": [],  # Removed "mediapipe.*" as it's already included
    "include_files": additional_datas,  # Include MediaPipe .tflite files
    "excludes": ["cv2.gapi.wip"],  # Exclude the problematic module
    "optimize": 2,  # Optimize bytecode
    "build_exe": "build/exe",  # Output directory
}

# ===========================  EXECUTABLE CONFIGURATION  ===========================
# Define base for GUI applications to hide the console
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use "Win32GUI" for GUI applications to hide the console

# Define executables
executables = [
    Executable(
        script="slouch.py",
        base=base,
        target_name="slouch.exe",
        icon=None  # You can specify an icon file here if desired
    ),
    Executable(
        script="watchdog.py",
        base="Win32GUI",  # Hide console for watchdog
        target_name="watchdog.exe",
        icon=None  # You can specify an icon file here if desired
    )
]

# ===========================  SETUP CONFIGURATION  ===========================
setup(
    name="SlouchLock",
    version="1.0",
    description="Slouch Prevention Application with Watchdog Mechanism",
    options={"build_exe": build_exe_options},
    executables=executables
)
