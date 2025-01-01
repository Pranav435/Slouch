# Slouch - ReadMe

## Overview

**Slouch** is a Windows-only posture monitoring application that encourages better sitting habits by detecting slouching using computer vision. It utilizes your webcam to monitor posture in real-time and take corrective action, such as locking input, when poor posture is detected. The tool features a 3-phase calibration wizard to personalize detection for each user.

---

## Features

1. **3-Phase Calibration Wizard**: Customizes posture detection based on your upright, look-down, and slouch postures.
2. **Real-Time Monitoring**: Detects slouching and takes action to prevent prolonged poor posture.
3. **Emergency Unlock**: Allows you to quickly restore input in case of false positives or errors.
4. **Admin Privileges Management**: Automatically relaunches with admin rights for full functionality.
5. **Configurable Parameters**: Tailors monitoring thresholds and calibration settings.

---

## Implementation

The Slouch algorithm involves multiple stages to ensure accurate posture detection and response. Here’s how it works:

### Calibration Phase
During calibration, the user is guided through three posture states:
1. **Upright**: The user sits straight, and metrics like slouch ratio and ear-shoulder-hip angles are recorded.
2. **Look Down**: The user lowers their head to mimic looking down, and metrics are captured.
3. **Slouch**: The user deliberately slouches, and the system measures the resulting changes in metrics.

These metrics are stored in a configuration file to create personalized thresholds for future posture monitoring.

### Monitoring Phase
In the monitoring phase, the algorithm uses:

1. **Slouch Ratio Calculation**: The nose-to-shoulder vertical difference is normalized by shoulder width to compute the slouch ratio. Higher values indicate forward tilt or slouching.
2. **Ear-Shoulder-Hip Angle**: The angles formed between the ear, shoulder, and hip are analyzed. Lower angles suggest leaning or hunching.
3. **Real-Time Classification**:
   - The slouch ratio and angles are compared against calibration thresholds to classify the posture as "Upright," "Look Down," or "Slouch."
   - If either metric exceeds the slouch threshold, the posture is classified as "Slouch."

### Response Mechanism
- **Input Blocking**: If slouching persists for a predefined number of frames, the application blocks keyboard and mouse input.
- **Beep Alerts**: Periodic beeps are triggered during slouching to remind the user to correct their posture.
- **Unlocking**: Input is restored when the user sits upright for a predefined number of consecutive frames.

### Emergency Unlock
An emergency unlock feature allows the user to bypass input blocking by pressing a shortcut (Ctrl+Alt+U). This ensures usability in unexpected scenarios.

The combination of these steps ensures real-time, personalized posture monitoring and correction.

---

## Installation

### Download the Release

1. Navigate to the [Releases](https://github.com/Pranav435/slouch/releases) page.
2. Download the latest version of **Slouch** as a pre-built `.exe` file.
3. Run the `.exe` file with administrative privileges to install Slouch to your system.

### Build From Source

1. Clone this repository:
   ```bash
   git clone https://github.com/Pranav435/slouch.git
   cd slouch
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python slouch.py
   ```

For building an installable `.exe`, use the included setup.py file:
```bash
python setup.py build
```
The resulting `.exe` file will be in the `build\exe` directory.

---

## Usage

### Running the Application

1. Launch **Slouch** with administrative privileges to enable input blocking.
2. The program will automatically start the calibration wizard if no prior calibration exists.

### Calibration Wizard

- Follow the on-screen instructions to complete three phases:
  1. Upright
  2. Look Down
  3. Slouch

### Monitoring Posture

- Once calibration is complete, **Slouch** will monitor your posture.
- The application will:
  - Alert you with a beep when slouching is detected.
  - Lock input if slouching persists.
  - Unlock input when you return to an upright posture.

### Emergency Unlock

- Press **Ctrl+Alt+U** to immediately restore input control.

---

## Configuration

Configuration files are stored in:
```plaintext
C:\ProgramData\Slouch
```

### Files
- **`config.json`**: Stores settings for monitoring thresholds and durations.
- **`three_phase_calibration.json`**: Stores calibration data for ratios and angles.

### Default Parameters
- **Phase Duration**: 4 seconds
- **Frames to Lock Input**: 15 consecutive slouch detections
- **Frames to Unlock Input**: 5 consecutive upright detections

You can edit these files to adjust thresholds and durations.

---

## Logging

Logs are saved in:
```plaintext
C:\ProgramData\Slouch\slouch_log.log
```
Use these logs to debug any issues or review posture monitoring activity.

---

## Known Issues

1. **Webcam Compatibility**: Ensure your webcam supports a resolution of at least 320x240.
2. **Lighting Conditions**: Poor or inconsistent lighting may affect posture detection accuracy.
3. **Admin Rights**: The application requires administrative privileges for full functionality.

---

## Shortcuts

- **Start Calibration**: Ctrl+N
- **Emergency Unlock**: Ctrl+Alt+U

---

## Contributions

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests on the [GitHub repository](https://github.com/Pranav435/slouch).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Special thanks to:
- Mediapipe for pose estimation.
- Open-source libraries like WxPython, OpenCV, and NumPy for making this project possible.

