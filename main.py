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
camera = cv.VideoCapture(0, cv.CAP_DSHOW)
if not camera.isOpened():
    print("Не удалось открыть камеру")
    exit()

def detect_chessboard(frame):    
        img = frame.copy()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Detecting edges
        canny_img = cv.Canny(gray_img, 100, 150)
        cv.imshow('edges', canny_img)

        #Widering edges
        kernel = np.ones((2,2), np.uint8)
        wider_img = cv.dilate(canny_img, kernel, iterations=1)
        cv.imshow('wider img', wider_img)

        threshold = 200
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

            if 4 <= len(approx) <= 6:
                cv.drawContours(black_img_squares, [approx], -1, 255, thickness=4)
                rect_count += 1
        cv.imshow('4-angled figures', black_img_squares)



        board_contours, hierarchy = cv.findContours(black_img_squares, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        black_img_2 = np.zeros_like(black_img_squares)

        square_centers = list()

        board_squared = canny_img.copy() # copying the board img

        # loop through contours and filter them by deciding if they are potential squares
        for contour in board_contours:
            if MIN_SQUARE_LENGTH**2 < cv.contourArea(contour) < MAX_SQUARE_LENGTH**2:

                # Approximate the contour to a simpler shape
                epsilon = 0.02 * cv.arcLength(contour, True) # Типо сглаживание
                approx = cv.approxPolyDP(contour, epsilon, True) # Оставляем только углы

                if len(approx) == 4:
                    pts = [pt[0].tolist() for pt in approx]
                    
                    #Сортируем вначале 2 правые точки, потом 2 левые
                    index_sorted = sorted(pts, key=lambda x: x[0], reverse=True)
                    
                    #Точку которая выше ставим на первый план (среди правых точек)
                    if index_sorted[0][1]< index_sorted[1][1]:
                        #Просто меняем местами
                        cur = index_sorted[0]
                        index_sorted[0] = index_sorted[1]
                        index_sorted[1] = cur

                    #Точку которая выше ставим на первый план (среди левых точек)
                    if index_sorted[2][1]> index_sorted[3][1]:
                        cur = index_sorted[2]
                        index_sorted[2] = index_sorted[3]
                        index_sorted[3] = cur
                    
                    # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                    pt1=index_sorted[0]
                    pt2=index_sorted[1]
                    pt3=index_sorted[2]
                    pt4=index_sorted[3]

                    # find rectangle that fits 4 point 
                    x, y, w, h = cv.boundingRect(contour)

                    center_x = (x+(x+w))/2
                    center_y = (y+(y+h))/2

                    # lengths of sides
                    l1 = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    l2 = math.sqrt((pt2[0] - pt3[0])**2 + (pt2[1] - pt3[1])**2)
                    l3 = math.sqrt((pt3[0] - pt4[0])**2 + (pt3[1] - pt4[1])**2)
                    l4 = math.sqrt((pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2)

                    # Create a list of lengths
                    lengths = [l1, l2, l3, l4]
                    
                    # Get the maximum and minimum lengths
                    max_length = max(lengths)
                    min_length = min(lengths)

                    # Check if this length values are suitable for a square , this threshold value plays crucial role for squares ,  
                    valid_square=True
                    if (max_length - min_length) <= 35: # 20 for smaller boards  , 50 for bigger , 35 works most of the time 
                        pass
                    else:
                        valid_square=False
                    
                    if valid_square:
                        square_centers.append([center_x, center_y, pt1, pt2, pt3, pt4])

                        cv.line(board_squared, pt1, pt2, (255, 255, 0), 7)
                        cv.line(board_squared, pt2, pt3, (255, 255, 0), 7)
                        cv.line(board_squared, pt3, pt4, (255, 255, 0), 7)
                        cv.line(board_squared, pt1, pt4, (255, 255, 0), 7)

                        # Draw only valid squares to "black_image_2"
                        cv.line(black_img_2, pt1, pt2, 255, 7)
                        cv.line(black_img_2, pt2, pt3, 255, 7)
                        cv.line(black_img_2, pt3, pt4, 255, 7)
                        cv.line(black_img_2, pt1, pt4, 255, 7)
                        

        cv.imshow(' 1 ',board_squared)
        cv.imshow(' 2 ',black_img_2)
        




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
