import cv2 as cv
import numpy as np
from ultralytics import YOLO
import chess
import time

# Class id to label mapping - adjust based on your model
CLASS_ID_TO_NAME = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}

# If your model includes chessboard detection, uncomment and adjust:
# CLASS_ID_TO_NAME[12] = 'chessboard'

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

def detect_objects(frame):
    """Detect all objects (pieces and possibly chessboard) using YOLO model"""
    if model is None:
        print("YOLO model not loaded")
        return [], None
    
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
    
    return detected_pieces, chessboard_bbox

def map_pieces_to_chess_coordinates(chessboard_bbox, detected_pieces):
    """Map detected pieces to chess coordinates based on chessboard bounding box"""
    if chessboard_bbox is None:
        return detected_pieces
    
    x1, y1, x2, y2 = chessboard_bbox
    board_width = x2 - x1
    board_height = y2 - y1
    
    # Calculate square size
    square_width = board_width / 8
    square_height = board_height / 8
    
    for piece in detected_pieces:
        center_x, center_y = piece['center']
        
        # Check if piece is within the chessboard bounds
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            # Calculate relative position within the board
            rel_x = center_x - x1
            rel_y = center_y - y1
            
            # Calculate chess coordinates
            col = int(rel_x / square_width)
            row = int(rel_y / square_height)
            
            # Ensure bounds
            col = max(0, min(7, col))
            row = max(0, min(7, row))
            
            # Convert to chess notation (a1, b1, etc.)
            file_letter = chr(97 + col)  # 'a' to 'h'
            rank_number = 8 - row        # 8 to 1
            
            chess_coordinate = f"{file_letter}{rank_number}"
            piece['chess_coordinate'] = chess_coordinate
            
            # Draw coordinate label
            cv.putText(frame, chess_coordinate, (int(center_x) + 10, int(center_y)), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            piece['chess_coordinate'] = None
    
    return detected_pieces

def draw_chess_grid(frame, chessboard_bbox):
    """Draw 8x8 grid on the detected chessboard"""
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
    
    # Draw coordinate labels
    for col in range(8):
        for row in range(8):
            file_letter = chr(97 + col)
            rank_number = 8 - row
            coord = f"{file_letter}{rank_number}"
            
            # Calculate position for label
            label_x = x1 + (board_width / 8) * col + 5
            label_y = y1 + (board_height / 8) * (row + 1) - 5
            
            cv.putText(frame, coord, (int(label_x), int(label_y)), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def create_chess_board_state(pieces_with_coordinates):
    """Create a chess board state from detected pieces"""
    board = chess.Board()
    board.clear()  # Clear the board
    
    for piece in pieces_with_coordinates:
        if piece.get('chess_coordinate') is None:
            continue
            
        try:
            square = chess.parse_square(piece['chess_coordinate'])
            piece_symbol = piece['piece_symbol']
            
            # Map piece symbols to chess pieces
            piece_map = {
                'P': chess.PAWN, 'N': chess.KNIGHT, 'B': chess.BISHOP,
                'R': chess.ROOK, 'Q': chess.QUEEN, 'K': chess.KING,
                'p': chess.PAWN, 'n': chess.KNIGHT, 'b': chess.BISHOP,
                'r': chess.ROOK, 'q': chess.QUEEN, 'k': chess.KING
            }
            
            if piece_symbol in piece_map:
                piece_type = piece_map[piece_symbol]
                color = chess.WHITE if piece_symbol.isupper() else chess.BLACK
                board.set_piece_at(square, chess.Piece(piece_type, color))
        except:
            continue
    
    return board

def main():
    """Main function to run flexible chess detection"""
    print("Starting flexible chess detection...")
    print("Press 'q' to quit")
    print("This will detect pieces and chessboard (if model supports it)")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Detect objects
        detected_pieces, chessboard_bbox = detect_objects(frame)
        
        # Map pieces to coordinates if chessboard detected
        if chessboard_bbox is not None:
            draw_chess_grid(frame, chessboard_bbox)
            detected_pieces = map_pieces_to_chess_coordinates(chessboard_bbox, detected_pieces)
            
            # Create chess board state
            chess_board = create_chess_board_state(detected_pieces)
            
            # Display results
            print(f"\nChessboard detected at: {chessboard_bbox}")
            print(f"Detected {len(detected_pieces)} pieces:")
            for piece in detected_pieces:
                if piece.get('chess_coordinate'):
                    print(f"  {piece['piece_name']} at {piece['chess_coordinate']} (confidence: {piece['confidence']:.2f})")
                else:
                    print(f"  {piece['piece_name']} at pixel coordinates ({piece['pixel_coordinates'][0]}, {piece['pixel_coordinates'][1]}) (confidence: {piece['confidence']:.2f})")
            
            print(f"\nChess board FEN: {chess_board.fen()}")
            print(f"Board state:\n{chess_board}")
        else:
            print(f"\nNo chessboard detected, showing {len(detected_pieces)} pieces with pixel coordinates:")
            for piece in detected_pieces:
                print(f"  {piece['piece_name']} at pixel coordinates ({piece['pixel_coordinates'][0]}, {piece['pixel_coordinates'][1]}) (confidence: {piece['confidence']:.2f})")
        
        # Show the frame with detections
        cv.imshow('Flexible Chess Detection', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


