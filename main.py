import cv2 as cv
import mss
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv

def video_capture():
    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)

            cv.imshow("Live", frame)

            if cv.waitKey(1000) & 0xFF == ord('q'):
                break
                
    cv.destroyAllWindows()

IMAGE_PATH = r'test-10.jpeg'

img = cv.imread(IMAGE_PATH)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Processing Image

#Threeshold (Each pizel will be exactly black or white)
#ret - выбранный порог от cv.THRESH_BINARY + cv.THRESH_OTSU
#otsu_binary - b&w img
ret, otsu_binary = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

#Canny (edges)
canny_img = cv.Canny(otsu_binary, 20, 255)

#Widering edges
kernel = np.ones((7,7), np.uint8)
dilation_img = cv.dilate(canny_img, kernel, iterations=1)



#Finding straight lines
# 1 - точность по ширине
# np.pi / 180 - точность по углу (1 градус)
#threshold = 500 - минимальное количество точек на линии, чтобы она считалась действительной
lines = cv.HoughLinesP(dilation_img, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)
'''
print("Линий найдено:", len(lines) if lines is not None else 0)
'''

black_img = np.zeros_like(dilation_img)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(black_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

#Widering
kernel = np.ones((3,3), np.uint8)
black_img_lined = cv.dilate(black_img, kernel, iterations=1)
'''
plt.figure(figsize=(9,7))
plt.title("All Lines (black_image)")
plt.imshow(black_img_lined, cmap="gray")
plt.show()
'''

# Look for valid squares and check if squares are inside of board
#contours
board_contours, hierarchy = cv.findContours(black_img_lined, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

black_img_2 = np.zeros_like(black_img_lined)

square_centers = list()

board_squared = canny_img.copy() # copying the board img

# loop through contours and filter them by deciding if they are potential squares
for contour in board_contours:
    if 2000 < cv.contourArea(contour) < 20000:

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
                cv.line(black_img_2, pt1, pt2, (255, 255, 0), 7)
                cv.line(black_img_2, pt2, pt3, (255, 255, 0), 7)
                cv.line(black_img_2, pt3, pt4, (255, 255, 0), 7)
                cv.line(black_img_2, pt1, pt4, (255, 255, 0), 7)
'''
plt.figure(figsize=(12,8))

plt.subplot(121)
plt.title("board_squared")
plt.imshow(board_squared,cmap="gray")

plt.subplot(122)
plt.title("black_image_2")
plt.imshow(black_img_2,cmap="gray")
plt.show()
'''

kernel = np.ones((7,7), np.uint8)
dilated_black_img = cv.dilate(black_img_2, kernel, iterations=1)


#finding the biggest contour
contours, _ = cv.findContours(dilated_black_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv.contourArea)
largest_contour_img = np.zeros_like(dilated_black_img)

cv.drawContours(largest_contour_img, largest_contour, -1, (255,255,255), 10)
'''
plt.figure(figsize=(9,7))
plt.title("Biggest contour")
plt.imshow(largest_contour_img)
plt.show()
'''

inside_squares = list()

for square in square_centers:
    point=(square[0],square[1])

    #Checking if it is inside
    distance = cv.pointPolygonTest(largest_contour, point, measureDist=False)

    if distance >= 0:
        inside_squares.append(square)
    else:
        continue

#print(len(inside_squares)) // 64

#sorting the squares

#sorting y
sorted_coordinates = sorted(inside_squares, key=lambda x: x[1], reverse=True)

#sorting x
groups = []
current_group = [sorted_coordinates[0]]

for coord in sorted_coordinates[1:]:
    if abs(coord[1] - current_group[-1][1]) < 50:
        current_group.append(coord)
    else:
        groups.append(current_group)
        current_group = [coord]

groups.append(current_group)

for group in groups:
    group.sort(key=lambda x: x[0])

sorted_coordinates = [coord for group in groups for coord in group]

def fill_gaps():
    global sorted_coordinates
    
    addition = 0

    for num in range(63):

        if num in[6,14,22,30,38,46,54]:
            if abs(sorted_coordinates[num][0]-sorted_coordinates[num+1][0])>250:

                x=sorted_coordinates[num][0] + abs(sorted_coordinates[num][0]-sorted_coordinates[num-1][0])
                y=sorted_coordinates[num][1]

                p1 = (sorted_coordinates[num][2][0] + abs(sorted_coordinates[num][2][0]-sorted_coordinates[num-1][2][0]), sorted_coordinates[num][2][1])
                p2 = (sorted_coordinates[num][3][0] + abs(sorted_coordinates[num][3][0]-sorted_coordinates[num-1][3][0]), sorted_coordinates[num][3][1])
                p3 = sorted_coordinates[num][3]
                p4 = sorted_coordinates[num][2]

                sorted_coordinates.insert(num+1, [x,y,p1,p2,p3,p4])
                print('first statement', num+2)
                continue

        elif num in [8,16,24,32,40,48,56]:
            if abs(sorted_coordinates[num][0]-sorted_coordinates[num-8][0])>50:

                x=sorted_coordinates[num-8][0] 
                y=sorted_coordinates[num+1][1] 
                     
                p1=sorted_coordinates[num-8][3]
                p2=(sorted_coordinates[num-8][3][0],sorted_coordinates[num+1][3][1])
                p3=(sorted_coordinates[num-8][4][0],sorted_coordinates[num+1][3][1])
                p4=sorted_coordinates[num-8][4]
                
                sorted_coordinates.insert(num,[x,y,p1,p2,p3,p4])
                print("second statement",num+1)
                continue

        elif abs(sorted_coordinates[num][1] - sorted_coordinates[num+1][1])< 50 :
            if sorted_coordinates[num+1][0] - sorted_coordinates[num][0] > 150:
                x=(sorted_coordinates[num+1][0] + sorted_coordinates[num][0])/2
                y=(sorted_coordinates[num+1][1] + sorted_coordinates[num][1])/2
                p1=sorted_coordinates[num+1][5]
                p2=sorted_coordinates[num+1][4]
                p3=sorted_coordinates[num][3]
                p4=sorted_coordinates[num][2]
                sorted_coordinates.insert(num+1,[x,y,p1,p2,p3,p4])
                print(f"third statement",num+2)
                addition+=1

    if addition!=0:
        fill_gaps()


if len(inside_squares)!=64:            
    fill_gaps() 


image = cv.imread(IMAGE_PATH)
corners_image=cv.cvtColor(image,cv.COLOR_BGR2RGB)


p1=sorted_coordinates[0][5]
p2=sorted_coordinates[7][2]
p3=sorted_coordinates[56][4]
p4=sorted_coordinates[63][3]

cv.circle(corners_image, (int(p1[0]),int(p1[1])), 12, (0,255,0), 8)
cv.circle(corners_image, (int(p2[0]),int(p2[1])), 12, (0,255,0), 8)
cv.circle(corners_image, (int(p3[0]),int(p3[1])), 12, (0,255,0), 8)
cv.circle(corners_image, (int(p4[0]),int(p4[1])), 12, (0,255,0), 8)

'''
plt.figure(figsize=(9,7))
plt.imshow(corners_image)
plt.show()
'''
#Coordinates to CSV file

with open("board-square-positions.csv", mode='w', newline='')as file:
    writer = csv.writer(file)

    writer.writerow(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

    for coordinate in sorted_coordinates:
        writer.writerow([coordinate[2][0], coordinate[2][1],
                        coordinate[3][0], coordinate[3][1],
                        coordinate[4][0], coordinate[4][1],
                        coordinate[5][0], coordinate[5][1]])

data = pd.read_csv("board-square-positions.csv")
img = cv.imread(IMAGE_PATH)

for i, row in data.iterrows():
    pts = []
    for j in range(0, 8, 2):
        pts.append((int(row[j]), int(row[j+1])))
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    cv.circle(img, (int(sorted_coordinates[i][0]),int(sorted_coordinates[i][1])), 3, (0,255,0), 3)
    cv.polylines(img,[pts],True,(255,255,255),thickness=8)

'''
plt.figure(figsize=(10,8))
plt.imshow(img)
plt.show()
'''

coordinates = pd.read.csv("board-sqiare-positions.csv")

#Dictionary for every cell's boundary coordinates
# 64 in total

coord_dict = {}

cell=1
for row in coordinates.values:
    coord_dict[cell]=[[row[0], row[1]], [row[2],row[3]],[row[4],row[5]], [row[6],row[7]]]
    cell+=1

print(coord_dict[1])
print(len(coord_dict))

# class values , these values are decided before training
names: ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'] # type: ignore
class_dict={0:'black-bishop',1:'black-king',2:'black-knight',3:'black-pawn',4: 'black-queen',5: 'black-rook',
            6:'white-bishop',7:'white-king',8: 'white-knight',9: 'white-pawn',10: 'white-queen',11:'white-rook'}

