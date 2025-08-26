#this code finds the board without using any Neural network models. (cant find the board when pieces are placed on it)


import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
import chess
import chess.engine
import time

INTERVAL = 10
LAST_CAPTURE_TIME = 0
MIN_SQUARE_LENGTH = 30
MAX_SQUARE_LENGTH = 50

MIN_BOARD_LENGTH = 240
MAX_BOARD_LENGTH = 400
# Class id to label mapping must match training
CLASS_ID_TO_NAME = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}

# Load YOLO model
try:
    model = YOLO('chess-model-yolov8m.pt')
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

camera = cv.VideoCapture(0, cv.CAP_DSHOW)
if not camera.isOpened():
    print("Не удалось открыть камеру")
    exit()

def detect_chessboard(frame):    
        img = frame.copy()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise from pieces
        blurred = cv.GaussianBlur(gray_img, (5, 5), 0)
        
        #Detecting edges with lower sensitivity to reduce piece interference
        canny_img = cv.Canny(blurred, 50, 100)
        cv.imshow('edges', canny_img)

        #Widering edges with larger kernel to connect board edges
        kernel = np.ones((3,3), np.uint8)
        wider_img = cv.dilate(canny_img, kernel, iterations=2)
        cv.imshow('wider img', wider_img)

        threshold = 350
        black_img_lines = np.zeros_like(wider_img)
        lines = cv.HoughLines(wider_img, 1, np.pi / 180, threshold=threshold)

        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(black_img_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv.imshow('Lines', black_img_lines)            
            
        
        contours, _ = cv.findContours(black_img_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Отрисовка прямоугольников вместо линий
        black_img_squares = np.ones_like(black_img_lines)

        rect_count = 0  # Считаем прямоугольники

        for contour in contours:
            area = cv.contourArea(contour)
            if area < 200 or area > 15000:
                continue  # фильтруем слишком маленькие/большие

            epsilon = 0.03 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            # Из всех многоугольников нужно теперь извлечь только квадраты


            def is_square(approx, min_side=MIN_SQUARE_LENGTH, max_side=MAX_SQUARE_LENGTH, angle_tolerance=20):
                # approx - массив из 4 точек контура
                if len(approx) != 4:
                    return False

                def length(p1, p2):
                    return np.linalg.norm(p1 - p2)

                pts = approx.reshape(4, 2)
                sides = [length(pts[i], pts[(i+1)%4]) for i in range(4)]

                # Проверяем длины сторон в заданном диапазоне
                for side in sides:
                    if side < min_side or side > max_side:
                        return False

                # Проверяем, что все стороны примерно равны (отклонение не более 15%)
                mean_side = np.mean(sides)
                for side in sides:
                    if abs(side - mean_side) > mean_side * 0.15:
                        return False

                # Проверяем углы, они должны быть примерно 90 градусов
                def angle(pt1, pt2, pt3):
                    vec1 = pt1 - pt2
                    vec2 = pt3 - pt2
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    return angle_deg

                angles = []
                for i in range(4):
                    ang = angle(pts[(i-1)%4], pts[i], pts[(i+1)%4])
                    angles.append(ang)

                for ang in angles:
                    if abs(ang - 90) > angle_tolerance:
                        return False

                return True

        squares = []
        for contour in contours:
                area = cv.contourArea(contour)
                if area < 150 or area > 20000:
                    continue

                epsilon = 0.03 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4 and is_square(approx):
                    squares.append(approx.reshape(4,2))
                    cv.drawContours(black_img_squares, [approx], -1, 255, thickness=4)
                    rect_count += 1
        cv.imshow('squares', black_img_squares)

        #finding the centers of the squares
        mid_squares = []

        if len(squares) == 0:
            print("No squares detected")
            return None

        for square in squares:
            sum_x = 0
            sum_y = 0
            for point in square:
                sum_x += point[0]
                sum_y += point[1]
            mid_x = sum_x / 4
            mid_y = sum_y / 4
            mid_squares.append((mid_x, mid_y))


        #finding the biggest contour 
        contours, _ = cv.findContours(wider_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            print('No chess board detected')
            return None

        largest_contour = max(contours, key=cv.contourArea)
        largest_contour_img = np.zeros_like(wider_img)

        cv.drawContours(largest_contour_img, largest_contour, -1, (255,255,255), 10)
        cv.imshow('board', largest_contour_img)

        # Find the actual corners of the chessboard using contour approximation
        epsilon = 0.02 * cv.arcLength(largest_contour, True)
        approx_corners = cv.approxPolyDP(largest_contour, epsilon, True)
        
        # Ensure we have exactly 4 corners
        if len(approx_corners) != 4:
            print("Could not detect exactly 4 corners of the chessboard")
            return None
            
        # Sort corners in order: top-left, top-right, bottom-right, bottom-left
        corners = approx_corners.reshape(4, 2)
        
        # Sort by y-coordinate first to separate top and bottom
        corners = corners[corners[:, 1].argsort()]
        top_corners = corners[:2]
        bottom_corners = corners[2:]
        
        # Sort top corners by x-coordinate
        top_corners = top_corners[top_corners[:, 0].argsort()]
        # Sort bottom corners by x-coordinate
        bottom_corners = bottom_corners[bottom_corners[:, 0].argsort()]
        
        # Reconstruct the ordered corners
        board_corners = np.array([
            top_corners[0],      # Top-left
            top_corners[1],      # Top-right
            bottom_corners[1],   # Bottom-right
            bottom_corners[0]    # Bottom-left
        ], dtype=np.float32)
        
        # Check which squares are inside the chess board using contour point test
        board_squares = []
        for square in mid_squares:
            square_x, square_y = square
            # Check if the square center is inside the board contour
            if cv.pointPolygonTest(largest_contour, (square_x, square_y), False) >= 0:
                board_squares.append(square)
        
        is_board = False
        if len(board_squares) > 20:  # Reduced threshold to be more tolerant
            is_board = True
            print(f"Chess board detected with {len(board_squares)} squares")
        else:
            print(f"Not enough squares detected: {len(board_squares)} (need > 20)")
            
        # we divide the board into 64 pieces with angle consideration
        if is_board == True:
            new_board = frame.copy()
            
            # Calculate the size of the board in the transformed space
            # We'll use a standard 8x8 grid size for the transformed image
            grid_size = 400  # Size of the transformed board image
            square_size = grid_size // 8
            
            # Define the destination corners for perspective transform
            dst_corners = np.array([
                [0, 0],                    # Top-left
                [grid_size, 0],            # Top-right
                [grid_size, grid_size],    # Bottom-right
                [0, grid_size]             # Bottom-left
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            perspective_matrix = cv.getPerspectiveTransform(board_corners, dst_corners)
            
            # Apply perspective transform to get the straightened board
            transformed_board = cv.warpPerspective(new_board, perspective_matrix, (grid_size, grid_size))
            
            # Draw the detected board corners on the original image
            for i, corner in enumerate(board_corners):
                cv.circle(new_board, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
                cv.putText(new_board, str(i), tuple(corner.astype(int)), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Create 8x8 grid of squares on the transformed board
            squares_grid = []
            for row in range(8):
                for col in range(8):
                    # Calculate square coordinates in transformed space
                    x1 = col * square_size
                    y1 = row * square_size
                    x2 = (col + 1) * square_size
                    y2 = (row + 1) * square_size
                    
                    # Extract square from the transformed board
                    square = transformed_board[y1:y2, x1:x2]
                    
                    # Calculate the corresponding coordinates in the original image
                    # Transform the square corners back to original image space
                    square_corners_transformed = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32)
                    
                    # Apply inverse perspective transform
                    inverse_matrix = cv.getPerspectiveTransform(dst_corners, board_corners)
                    square_corners_original = cv.perspectiveTransform(
                        square_corners_transformed.reshape(-1, 1, 2), inverse_matrix
                    ).reshape(-1, 2)
                    
                    # Draw the transformed square on the original image
                    square_corners_original_int = square_corners_original.astype(int)
                    cv.polylines(new_board, [square_corners_original_int], True, (0, 255, 0), 2)
                    
                    # Add square coordinates and content to the grid
                    squares_grid.append({
                        'position': (row, col),
                        'coordinates_transformed': (x1, y1, x2, y2),
                        'coordinates_original': square_corners_original.tolist(),
                        'content': square
                    })
                
            # Display both the original image with drawn squares and the transformed board
            cv.imshow('Divided Chess Board (Original)', new_board)
            cv.imshow('Transformed Chess Board', transformed_board)
            print(f"Board divided into 64 squares with angle consideration. Grid size: {len(squares_grid)} squares")
            
            return squares_grid
        else:
            print("no chessboard detected, trying again")
            return None

while True:
    ret, frame = camera.read()
    if not ret:
        print("Не удалось считать кадр")
        break
    
    now = time.time()
    if now - LAST_CAPTURE_TIME >= INTERVAL:
        cv.imshow('Screenshot',frame)
        detect_chessboard(frame)
        LAST_CAPTURE_TIME = now

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()


