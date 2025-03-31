# Eye Tracker Application

A Python-based eye tracking application that uses computer vision to track user gaze and analyze visual attention patterns. The application supports two modes: YouTube video analysis and product layout analysis.

## Features

- Real-time eye tracking using MediaPipe Face Mesh
- Calibration system for accurate gaze mapping
- Two analysis modes:
  - YouTube Video Analysis: Tracks gaze while watching YouTube videos
  - Product Layout Analysis: Analyzes attention patterns on product layouts
- Heatmap visualization of gaze patterns
- Cross-platform support (Windows, macOS, Linux)
- Smooth gaze point tracking with adaptive calibration

## Prerequisites

- Python 3.8 or higher
- Webcam with good resolution (720p or higher recommended)
- Sufficient lighting for face detection
- Internet connection (for YouTube mode)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eyetracker.git
cd eyetracker
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Choose your analysis mode:
   - Option 1: YouTube Video Analysis
   - Option 2: Product Layout Analysis

3. Follow the calibration process:
   - Look at each red dot that appears on the screen
   - Press SPACE to capture your gaze for each point
   - Complete all 9 calibration points

4. For YouTube mode:
   - Watch the video while the application tracks your gaze
   - Press SPACE to end tracking and view the heatmap

5. For Product Layout mode:
   - Look at the product layout while the application tracks your gaze
   - Press SPACE to end tracking and view the heatmap and attention analysis

## Important Notes

- Ensure good lighting conditions for accurate face detection
- Keep your face centered in the camera view
- Maintain a consistent distance from the camera during calibration
- For macOS users: Grant camera access permission to your Terminal/Python

## Troubleshooting

1. Camera not detected:
   - Check if your webcam is properly connected
   - Ensure camera permissions are granted
   - Try using a different camera index in the code

2. Face detection issues:
   - Improve lighting conditions
   - Ensure your face is clearly visible
   - Check if you're too close or too far from the camera

3. Calibration problems:
   - Make sure to complete all calibration points
   - Keep your head position consistent
   - Try recalibrating if tracking seems inaccurate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for face mesh detection
- OpenCV for computer vision operations
- PyTubeFix for YouTube video handling

