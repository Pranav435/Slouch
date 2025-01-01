Slouch - ReadMe

Overview

Slouch is a Windows-only posture monitoring application that encourages better sitting habits by detecting slouching using computer vision. It utilizes your webcam to monitor posture in real-time and take corrective action, such as locking input, when poor posture is detected. The tool features a 3-phase calibration wizard to personalize detection for each user.

Features

3-Phase Calibration Wizard: Customizes posture detection based on your upright, look-down, and slouch postures.

Real-Time Monitoring: Detects slouching and takes action to prevent prolonged poor posture.

Emergency Unlock: Allows you to quickly restore input in case of false positives or errors.

Admin Privileges Management: Automatically relaunches with admin rights for full functionality.

Configurable Parameters: Tailors monitoring thresholds and calibration settings.

Installation

Download the Release

Navigate to the Releases page.

Download the latest version of Slouch as a pre-built .exe file.

Run the .exe file with administrative privileges.

Build From Source

Clone this repository:

git clone https://github.com/yourusername/slouch.git
cd slouch

Install the required Python packages:

pip install -r requirements.txt

Run the script:

python slouch.py

For building an installable .exe, use PyInstaller:

pyinstaller --onefile --noconsole slouch.py

The resulting .exe file will be in the dist directory.

Usage

Running the Application

Launch Slouch with administrative privileges to enable input blocking.

The program will automatically start the calibration wizard if no prior calibration exists.

Calibration Wizard

Follow the on-screen instructions to complete three phases:

Upright

Look Down

Slouch

Monitoring Posture

Once calibration is complete, Slouch will monitor your posture.

The application will:

Alert you with a beep when slouching is detected.

Lock input if slouching persists.

Unlock input when you return to an upright posture.

Emergency Unlock

Press Ctrl+Alt+U to immediately restore input control.

Configuration

Configuration files are stored in:

C:\ProgramData\Slouch

Files

config.json: Stores settings for monitoring thresholds and durations.

three_phase_calibration.json: Stores calibration data for ratios and angles.

Default Parameters

Phase Duration: 4 seconds

Frames to Lock Input: 15 consecutive slouch detections

Frames to Unlock Input: 5 consecutive upright detections

You can edit these files to adjust thresholds and durations.

Logging

Logs are saved in:

C:\ProgramData\Slouch\slouch_log.log

Use these logs to debug any issues or review posture monitoring activity.

Known Issues

Webcam Compatibility: Ensure your webcam supports a resolution of at least 320x240.

Lighting Conditions: Poor or inconsistent lighting may affect posture detection accuracy.

Admin Rights: The application requires administrative privileges for full functionality.

Shortcuts

Start Calibration: Ctrl+N

Emergency Unlock: Ctrl+Alt+U

Contributions

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests on the GitHub repository.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Special thanks to:

Mediapipe for pose estimation.

Open-source libraries like WxPython, OpenCV, and NumPy for making this project possible.

