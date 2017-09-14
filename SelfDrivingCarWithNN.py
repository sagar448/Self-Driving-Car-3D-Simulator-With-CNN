import cv2
import mss
import numpy as np
from collections import deque
import pyautogui as p

def GetSlopesOfLanes(OrgImage):
    #Convert the original image to gray
    GrayImg = cv2.cvtColor(OrgImage, cv2.COLOR_BGR2GRAY)
    #Detect edges in the image
    #150 is the max val, any edges above the intensity gradient of 150 are edges
    #65 is the lowest intensity gradient, anything below is not an edge
    imageWithEdges = cv2.Canny(GrayImg, threshold1=100, threshold2=200)
    #Apply blurring
    #The 5x5 is the gaussianblur kernel convolved with image
    #The 0 is the sigmaX and SigmaY standard deviation usually taken as 0
    blurredImg = cv2.GaussianBlur(imageWithEdges, (5, 5), 0)
    #These are the points of our trapezoid/hexagon that we crop out 
    points = np.array([[0, 500],[0, 250], [280, 200], [320, 200], [800, 350], [800, 500]])
    #Now we calculate the region of interest
    #We first create a mask (a blank black array same size as our image)
    mask = np.zeros_like(blurredImg)
    #Then we fill the mask underneath the points(inside the polygon) we defined
    #with color 255(This is the part of the image we want to keep)
    cv2.fillPoly(mask, [points], 255)
    #this bitwise and function basically combines the two images
    #The coloured bit where the pixels had a value is kept while the
    #top bit is removed
    croppedImg = cv2.bitwise_and(blurredImg, mask)
    #Basically the accumulation of the most imperfect edges with the minimum
    #length being defined by 180
    #Thickness of the lines is 5
    lines = cv2.HoughLinesP(croppedImg, 1, np.pi/180, 180, np.array([]), 180, 5)
    #Now we need to find the slope, intercept and length of each of our detected lines
    left_lines = []
    length_left = []
    right_lines = []
    length_right = []
    #We may not always detect a line that is why we do try/except statement
    try:
        for line in lines:
            #Coordinates of a single line
            for x1, y1, x2, y2 in line:
                #We dont want a vertical line or a horizontal line
                if x1==x2 or y1==y2:
                    continue
                #Slope formula
                slope = (y2-y1)/(x2-x1)
                #Intercept
                intercept = y1 - slope*x1
                #Length
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                #Y is reversed in images therefore a negative slope is a left line not right
                if slope<0:
                    left_lines.append((slope, intercept))
                    length_left.append((length))
                else:
                    right_lines.append((slope, intercept))
                    length_right.append((length))
    except:
        print("Nothing detected")
        pass
    #Now we have collected our similar lines into right and left lists
    #Now we can convert them into lanes by dot product all the similar lines with lengths
    #The longer lines are weighted more therefore affect the lanes more
    #Then we normalise them by dividing by sum of the lengths(sort of like averaginng)
    left_lane  = np.dot(length_left,  left_lines) /np.sum(length_left)  if len(length_left) >0 else None
    right_lane = np.dot(length_right, right_lines)/np.sum(length_right) if len(length_right)>0 else None
    return OrgImage

def DrawLines(OrgImg, linesList):
    #It is not always necessary for us to get lines back therefore
    #we put it in a try/except statement
    try:
        #iterate over each line in the lines array
        for line in lines:
            #The only list in line is filled with 4 coordinates/points
            points = line[0]
            #first parameter is the image
            #Next two parameters are points of one line(coordinates, x,y)
            #Next parameter is the color and finally, 3 is the thickness
            cv2.line(OrgImg, (points[0], points[1]), (points[2], points[3]), [255, 255, 255], 3)
    except:
        pass

sct = mss.mss()
while True:
    game = {'top': 240, 'left': 0, 'width': 580, 'height': 340}
    gameImg = np.array(sct.grab(game))
    gameImg = cv2.resize(gameImg, (600, 400))
    #new_img, left_slope, right_slope = process_img(img)
    #img = GetSlopesOfLanes(gameImg)
    print(gameImg.shape)
    cv2.imshow('window', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
