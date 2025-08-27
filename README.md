# Chess Board and Piece Detection System

This project uses a trained YOLO model to detect chess pieces and chessboards, mapping pieces to board coordinates. It includes multiple detection modes to work with different model configurations.

## Features

- **Chess Board Detection**
- **Piece Detection**: Uses YOLO model to detect all 12 types of chess pieces (6 white, 6 black) (SOON)
- **Coordinate Mapping**: Maps detected pieces to chess coordinates (a1, b1, etc.)
- **Real-time Processing**: Processes video feed from camera in real-time
- **Visual Feedback**: Shows bounding boxes, labels, and coordinate information
- **Flexible Detection**: Works with or without chessboard detection in the model

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Full System with Chessboard Detection (`board_finder.py`)

Run the comprehensive system that expects both chessboard and piece detection:

```bash
python board_finder.py
```

This system:
- Expects the YOLO model to detect both chessboard and pieces(soon)
- Creates a perspective transform to normalize the board
- Maps detected pieces to chess coordinates (a1, b1, etc.)
- Displays the chess board state in FEN notation
- Shows multiple visualization windows

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
2. **Model not loading**: Ensure the `chess-model-yolov8m.pt` file is in the project directory
3. **Poor detection**: Improve lighting and ensure the chess board is clearly visible
4. **Chessboard not detected**: 
   - Check if your model was trained to detect chessboards
   - Adjust the class mapping in the script if needed

## File Structure

```
chess/
├── board_finder.py         # Original computer vision board detection
├── test.py                 # Camera test utility
├── chess-model-yolov8m.pt  # Trained YOLO model
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dependencies

- OpenCV (cv2)
- NumPy
- Ultralytics (YOLO)
- Python-chess
- Matplotlib
- PyTorch
- TorchVision

