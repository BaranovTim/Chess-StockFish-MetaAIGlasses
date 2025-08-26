import cv2 as cv
import numpy as np
from ultralytics import YOLO
import chess
import time

# Class id to label mapping must match training
CLASS_ID_TO_NAME = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook',
    12: 'chessboard'  # Add chessboard class if your model includes it
}

# Chess piece mapping for coordinates
PIECE_MAPPING = {
    'black-bishop': 'b', 'black-king': 'k', 'black-knight': 'n', 'black-pawn': 'p', 'black-queen': 'q', 'black-rook': 'r',
    'white-bishop': 'B', 'white-king': 'K', 'white-knight': 'N', 'white-pawn': 'P', 'white-queen': 'Q', 'white-rook': 'R'
}

# Load YOLO model
try:
    model = YOLO('chess-model-yolov8m.pt')
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Initialize camera
camera = cv.VideoCapture(0, cv.CAP_DSHOW)
if not camera.isOpened():
    print("Failed to open camera")
    exit()

def detect_chessboard_and_pieces_simple(frame):
    """Detect chess board and pieces using YOLO model"""
    if model is None:
        print("YOLO model not loaded")
        return None, []
    
    # Run YOLO detection
    results = model(frame)
    
    detected_pieces = []
    chessboard_bbox = None
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter by confidence
                if confidence < 0.5:
                    continue
                
                class_name = CLASS_ID_TO_NAME.get(class_id, f"unknown-{class_id}")
                
                # Check if this is a chessboard detection
                if class_name == 'chessboard':
                    chessboard_bbox = (x1, y1, x2, y2)
                    # Draw chessboard bounding box
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                    cv.putText(frame, f"Chessboard ({confidence:.2f})", (int(x1), int(y1) - 10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    # This is a chess piece
                    piece_symbol = PIECE_MAPPING.get(class_name, '?')
                    
                    # Calculate center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detected_pieces.append({
                        'piece_name': class_name,
                        'piece_symbol': piece_symbol,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'pixel_coordinates': (int(center_x), int(center_y))
                    })
                    
                    # Draw bounding box and label for pieces
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name} ({int(center_x)}, {int(center_y)})"
                    cv.putText(frame, label, (int(x1), int(y1) - 10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw center point
                    cv.circle(frame, (int(center_x), int(center_y)), 3, (255, 0, 0), -1)
    
    return chessboard_bbox, detected_pieces

def draw_simple_grid(frame, chessboard_bbox):
    """Draw simple 8x8 grid on the detected chessboard"""
    if chessboard_bbox is None:
        return
    
    x1, y1, x2, y2 = chessboard_bbox
    board_width = x2 - x1
    board_height = y2 - y1
    
    # Draw vertical lines
    for i in range(1, 8):
        x = x1 + (board_width / 8) * i
        cv.line(frame, (int(x), int(y1)), (int(x), int(y2)), (0, 255, 255), 1)
    
    # Draw horizontal lines
    for i in range(1, 8):
        y = y1 + (board_height / 8) * i
        cv.line(frame, (int(x1), int(y)), (int(x2), int(y)), (0, 255, 255), 1)

def main():
    """Main function to run simple chess board and piece detection"""
    print("Starting simple chess board and piece detection...")
    print("Press 'q' to quit")
    print("This will show detected chessboard and pieces with their pixel coordinates")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Detect chessboard and pieces
        chessboard_bbox, detected_pieces = detect_chessboard_and_pieces_simple(frame)
        
        # Draw grid if chessboard detected
        if chessboard_bbox is not None:
            draw_simple_grid(frame, chessboard_bbox)
            print(f"\nChessboard detected at: {chessboard_bbox}")
        
        # Display results
        if detected_pieces:
            print(f"Detected {len(detected_pieces)} pieces:")
            for piece in detected_pieces:
                print(f"  {piece['piece_name']} at pixel coordinates ({piece['pixel_coordinates'][0]}, {piece['pixel_coordinates'][1]}) (confidence: {piece['confidence']:.2f})")
        else:
            print("No pieces detected")
        
        # Show the frame with detections
        cv.imshow('Simple Chess Detection', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
