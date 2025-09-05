# Chess Board and Piece Detection System


## Features

<<<<<<< HEAD
- **Chess Board Detection**
- **Piece Detection**: Uses YOLO model to detect all 12 types of chess pieces (6 white, 6 black) (SOON)
=======
>>>>>>> 235339f (removed some pointless files)
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

<<<<<<< HEAD
=======
### Option 2: Simple Detection (`simple_detection.py`)

Run the simpler system that detects both chessboard and pieces:

```bash
python simple_detection.py
```

This system:
- Detects chessboard and pieces using the YOLO model
- Shows pixel coordinates for each detected piece
- Draws a grid overlay on the detected chessboard
- Provides real-time visual feedback

### Option 3: Flexible Detection (`flexible_detection.py`) - **Recommended**

Run the flexible system that works with any model configuration:

```bash
python flexible_detection.py
```

This system:
- Works whether your model detects chessboard or not
- Automatically adapts to available detections
- Maps pieces to chess coordinates when chessboard is detected
- Falls back to pixel coordinates when chessboard is not detected
- Most robust option for different model configurations

## Model Configuration

### If Your Model Includes Chessboard Detection

If your YOLO model was trained to detect chessboards, the class mapping should include:

```python
CLASS_ID_TO_NAME = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook',
    12: 'chessboard'  # Add this if your model detects chessboards
}
```

### If Your Model Only Detects Pieces

If your model only detects pieces, use the standard mapping:

```python
CLASS_ID_TO_NAME = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}
```


**Chessboard (if model supports it):**
- Complete chess board detection

## Output Information

### With Chessboard Detection
- Chessboard bounding box coordinates
- Number of detected pieces
- Piece names and chess coordinates (a1, b1, etc.)
- FEN notation of the board state
- Visual board representation with grid overlay

### Without Chessboard Detection
- Number of detected pieces
- Piece names and pixel coordinates
- Confidence scores for each detection

>>>>>>> 235339f (removed some pointless files)
## Controls

- Press 'q' to quit the application
- The system processes frames at regular intervals (10 seconds by default)

## Camera Setup

The system uses the default camera (index 0). Make sure:
- Your camera is connected and working
- The chess board is clearly visible
- Good lighting conditions for better detection

<<<<<<< HEAD
## Troubleshooting

1. **Camera not opening**: Check if your camera is connected and not being used by another application
2. **Model not loading**: Ensure the `chess-model-yolov8m.pt` file is in the project directory
3. **Poor detection**: Improve lighting and ensure the chess board is clearly visible
4. **Chessboard not detected**: 
   - Check if your model was trained to detect chessboards
   - Adjust the class mapping in the script if needed

=======
>>>>>>> 235339f (removed some pointless files)
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
<<<<<<< HEAD

=======
>>>>>>> 235339f (removed some pointless files)
