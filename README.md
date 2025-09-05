# Chess Board and Piece Detection System


## Features

- **Real-time Processing**: Processes video feed from camera in real-time
- **Visual Feedback**: Shows bounding boxes, labels, and coordinate information

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Full System with Chessboard Detection (`board_finder.py`)

Run the comprehensive system that expects chessboard detection:

python board_finder.py


## Output Information

### With Chessboard Detection
- Chessboard bounding box coordinates
- Number of detected squares
- Visual board representation with grid overlay

## Controls

- Press 'q' to quit the application
- The system processes frames at regular intervals (10 seconds by default)

## Camera Setup

The system uses the default camera (index 0). Make sure:
- Your camera is connected and working
- The chess board is clearly visible
- Good lighting conditions for better detection

## Troubleshooting

1. **Camera not opening**: Check if your camera is connected and not being used by another application
2. **Poor detection**: Improve lighting and ensure the chess board is clearly visible
3. **Chessboard not detected**: 
   - Check if your model was trained to detect chessboards
   - Adjust the class mapping in the script if needed

## File Structure

```
chess/
├── board_finder.py         # Original computer vision board detection
├── test.py                 # Camera test utility
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dependencies

- OpenCV (cv2)
- NumPy
- Python-chess
- Matplotlib
- PyTorch
- TorchVision