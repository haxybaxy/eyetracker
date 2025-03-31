# Eye Tracker Application

A Python-based eye tracking application that uses computer vision to track user gaze and analyze visual attention patterns. The application supports multiple modes: YouTube video analysis and product layout analysis, with both default and custom content options.

## Features

- Real-time eye tracking using MediaPipe Face Mesh
- Advanced calibration system with 9-point grid for accurate gaze mapping
- Multiple analysis modes:
  - YouTube Video Analysis (Default Video)
  - Product Layout Analysis (Default Image)
  - Custom YouTube Video Analysis
  - Custom Photo Analysis
- Real-time gaze point visualization with smoothing
- Gaze trail replay with time-compressed playback
- Heatmap visualization of gaze patterns
- Product attention analysis with region tracking
- Cross-platform support (Windows, macOS, Linux)
- Adaptive calibration based on inter-eye distance
- Data export capabilities for further analysis
- Customizable visualization options

## Prerequisites

- Python 3.8 or higher
- Webcam with good resolution (720p or higher recommended)
- Sufficient lighting for face detection
- Internet connection (for YouTube mode)
- Minimum 4GB RAM (8GB recommended)
- Modern GPU (optional, but recommended for better performance)

## Dependencies

The application requires the following Python packages with minimum versions:
- numpy >= 1.24.0: For numerical computations
- opencv-python >= 4.8.0: For computer vision operations
- mediapipe >= 0.10.0: For face mesh detection
- screeninfo >= 0.8.0: For screen resolution information
- scipy >= 1.11.0: For scientific computing
- pytubefix >= 4.0.0: For YouTube video handling
- matplotlib >= 3.7.0: For data visualization
- seaborn >= 0.12.0: For statistical visualizations
- pandas >= 2.0.0: For data manipulation and analysis
- pillow >= 10.0.0: For image processing

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
   - Option 1: YouTube Video Analysis (Default Video)
   - Option 2: Product Layout Analysis (Default Image)
   - Option 3: Custom YouTube Video Analysis
   - Option 4: Custom Photo Analysis

3. Follow the calibration process:
   - Look at each red dot that appears on the screen
   - Press SPACE to capture your gaze for each point
   - Complete all 9 calibration points
   - Keep your head position consistent during calibration

4. For YouTube modes:
   - For custom YouTube mode, enter the video URL when prompted
   - Watch the video while the application tracks your gaze
   - Press SPACE to end tracking
   - View the replay with gaze trail
   - View the final heatmap visualization
   - Export data for further analysis (optional)

5. For Product Layout modes:
   - For custom photo mode, provide the path to your image
   - Look at the product layout while the application tracks your gaze
   - Press SPACE to end tracking
   - View the replay with gaze trail
   - View the final heatmap visualization
   - Review attention analysis results for different product regions
   - Export gaze data and heatmap (optional)

## Important Notes

- Ensure good lighting conditions for accurate face detection
- Keep your face centered in the camera view
- Maintain a consistent distance from the camera during calibration
- For macOS users: Grant camera access permission to your Terminal/Python
- For optimal performance, close other resource-intensive applications
- Regular calibration is recommended for accurate tracking
- The application uses a 9-point calibration grid for improved accuracy
- Gaze tracking includes smoothing for more stable visualization
- The replay feature compresses the viewing time for better analysis

## Troubleshooting

1. Camera not detected:
   - Check if your webcam is properly connected
   - Ensure camera permissions are granted
   - Try using a different camera index in the code
   - Verify webcam drivers are up to date

2. Face detection issues:
   - Improve lighting conditions
   - Ensure your face is clearly visible
   - Check if you're too close or too far from the camera
   - Try adjusting camera position or angle

3. Calibration problems:
   - Make sure to complete all calibration points
   - Keep your head position consistent
   - Try recalibrating if tracking seems inaccurate
   - Ensure stable lighting during calibration

4. Performance issues:
   - Check system resource usage
   - Reduce screen resolution if needed
   - Update graphics drivers
   - Consider using a dedicated GPU

5. YouTube video issues:
   - Check your internet connection
   - Verify the video URL is valid and accessible
   - Ensure the video is not age-restricted or private
   - Try using a different video if problems persist

## Data Export

The application supports exporting:
- Raw gaze data (CSV format)
- Heatmap visualizations (PNG format)
- Statistical analysis reports (PDF format)
- Attention analysis results for product layouts
- Gaze trail replay data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Before submitting:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for face mesh detection
- OpenCV for computer vision operations
- PyTubeFix for YouTube video handling
- Contributors and maintainers of all dependencies

## Support

For support, please:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue if needed
4. Contact the maintainers

