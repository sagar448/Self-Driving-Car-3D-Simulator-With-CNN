# Self-Driving-Car-3D-Simulator-With-CNN
<p>
  <img align="left" width="425" height="400" src="https://github.com/sagar448/Self-Driving-Car-3D-Simulator-With-CNN/blob/master/src/3D%20Car%20Simulator.png">
  <img align="right" width="420" height="400" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Python.svg/2000px-Python.svg.png">
</p>
<p>
  <img align="center" width="600" height="7" src="http://getthedrift.com/wp-content/uploads/2015/06/White-Space.png">
</p>

## Introduction

Some point in our life as programmers we all wonder how a self driving car is actually made. I went through the same phase and so here it is, a very simple DIGITAL self driving car controlled using Python with a Reinforcement Q-learning algorithm as well as a Convolutional Neural Network.

You can essentially apply this to any game, the algorithm can be adapted and the reward rules can be changed to allow for different outcomes. I will go through the code step by step explaining what each line does and once you've mastered it you can go ahead fork the code and do as you wish.

```
Note: You need to have sufficient knowledge about Reinforcment learning before progressing, this tutorial 
only explains the code it does not go into the theoretical details
The links below help explain the theoretical details:

```

We will be using Keras to make the magic happen with Tensorflow backend. Assuming you are familiar with Keras and Tensorflow and have them installed we can start!

```
Note: Check my other gits for brief explanation on Keras and other simple algorithms such as the CNN 
and RNN if you are unfamiliar with Keras/Tensorflow!
```

## My Setup

In order to detect lanes, we need to send frames of our game to the algorithm for processing. I used a library called [mss](https://pypi.python.org/pypi/mss/)(MultipleScreenShot), it allows the users to take quick screenshots with almost minimal effect in FPS.
Unfortunately, it takes the screen shot of the entire screen if coordinates aren't specified, therefore in order to just get the frames of the game, the game needs to be properly positioned. 

The picture below depicts my environment.

<p align="center">
  <img width="800" height="600" src="https://github.com/sagar448/Self-Driving-Car-3D-Simulator-With-CNN/blob/master/src/Environment.png">
</p>
<p>
  <img width="600" height="4" src="http://getthedrift.com/wp-content/uploads/2015/06/White-Space.png">
</p>
You can set it up anyway you want but make sure to change the coordinates of the ScreenShot module so it only covers the game area.
Before we start the implementation it is a good idea to have the code open on the side as the comments have details you wouldn't want to miss.

## Implementation

### Imports
```python
1     import cv2
2     import mss
3     import numpy as np
4     from keras.models import Sequential
5     from keras.layers import Dense, Flatten
6     from keras.optimizers import SGD
7     from keras.layers.convolutional import Conv2D
8     import pyautogui as p
9     import random
10    import time
```
We start by importing a couple libraries. 
In order we import OpenCV, our Mss library, Numpy for computation, Keras for our CNN, Pyautogui to control our keyboard, Random and finally Time for delay purposes.

### Detecting Lanes
```python
1     #Function calculates the lanes
2     def CalculateLanes(OrgImage):
3         errors = False
4         #Since our game has yellow lanes, we can detect a specific color
5         #keep that color, and get rid of everything else to make it easier
6         #to detect the yellow lanes
7         #So we convert our image to the HSL color scheme
8         HSLImg = cv2.cvtColor(OrgImage, cv2.COLOR_BGR2HLS)
9         #The lower and upper arrays define boundaries of the BGR color space
10        #BGR because OpenCV represents images in Numpy in reverse order
11        #So for our yellow color we say that our pixels color that are yellow will be
12        # R>= 100, B >= 0, G>=10 (lower limit), R<=255, B<=255, G<=40
13        lower = np.uint8([ 10,   0, 100])
14        upper = np.uint8([ 40, 255, 255])
15        #inRange basically finds the color we want in the HLSImg with the lower and upper
16        #boundaries(the ranges)
17        yellow_mask = cv2.inRange(HSLImg, lower, upper)
18        #We then apply this mask to our original image, and this returns an image showing
19        #only the pixels that fall in the range of that mask
20        YellowImg = cv2.bitwise_and(OrgImage, OrgImage, mask=yellow_mask)
21        #Convert the original image to gray
22        GrayImg = cv2.cvtColor(YellowImg, cv2.COLOR_BGR2GRAY)
23        #Apply blurring
24        #The 5x5 is the gaussianblur kernel convolved with image
25        #The 0 is the sigmaX and SigmaY standard deviation usually taken as 0
26        blurredImg = cv2.GaussianBlur(GrayImg, (5, 5), 0)
27        #Detect edges in the image
28        #700 is the max val, any edges above the intensity gradient of 700 are edges
29        #200 is the lowest intensity gradient, anything below is not an edge
30        imageWithEdges = cv2.Canny(blurredImg, threshold1=200, threshold2=700)
31        #These are the points of our trapezoid/hexagon that we crop out 
32        points = np.array([[0, 310],[0, 300], [220, 210], [380, 210], [600, 300], [600, 310]])
33        #Now we calculate the region of interest
35        #We first create a mask (a blank black array same size as our image)
36        mask = np.zeros_like(imageWithEdges)
37        #Then we fill the mask underneath the points(inside the polygon) we defined
38        #with color white (255)(This is the part of the image we want to process)
39        cv2.fillPoly(mask, [points], 255)
40        #this bitwise and function basically combines the two images
41        #The coloured bit where the pixels had a value 255 is kept while the
42        #top bit is removed (which is 0)
43        croppedImg = cv2.bitwise_and(blurredImg, mask)
```
**Line 2** Our parameter being our screen shot (OrgImage)

**Line 3** We initialise our variable errors to False indicating that currently we have no errors produced

**Line 8** The lanes in the game are yellow and so we convert our image to the HSL color space in order to enhance our lanes.
They were not very clear in the RGB space, therefore HSL was used.

**Line 13&14** We define our upper and lower limit of the color space. Although the boundaries are given in terms of RGB it is actually in HSL. The comments are in RGB to make it easier to understand. Those limits represent the region where the color yellow falls within. Therefore, we use those limits so we can seek out a similar color in our image.

**Line 17** Now we apply the limits to our HSL image. It seeks out the color yellow and sets the rest of the pixels of the image to 0. We have essentially created a mask. An area where relevant pixels keep their values and the ones not needed are set to 0. 

**Line 20** The bitwise_and function basically takes a look at the pixel values and if the pixel value in the mask and the pixel value in the image have the same value they are kept, if they are different then it is set to 0. We are left with a image with only yellow region visible.

**Line 22** Now we convert our image to grayscale. We do this in order to make the edge detection more accurate. The canny edge detection function used later on essentially measures magnitude of pixel intensity changes. Therefore if we have colors that are similar to each other there isn't a big change in pixel intensity and it might not be considered an edge and ofcourse grayscale images are less computation heavy.

**Line 26** We now apply a gaussian blur. We do this in order to get rid of rough edges. Some realistic games or even in real life there are cracks on the road that might be considered something of interest so in order to get rid of the "noisy edges" we apply a blur

**Line 30** Now we finally apply the edge detection function. We have thresholds that identify what can and cannot be considered an edge.

**Line 32** We don't want all the edges detected in the image. We only want those that concern the lanes. So we create a region of interest, coordinates. 

**Line 36** We create an empty black mask with the same space dimension as our image.

**Line 39** The points that defined our ROI (Polygon), we fill the mask with the color white (255) inside that shape.

**Line 43** Finally we take our blurred image and we apply our mask to it. So the white region of our mask is replaced with our image while the rest is black (not used)

Great now we've managed to narrow down our edges to the region that we are interested in. Thats most of the processing done. We now want to get the appropriate lines and combine them into lanes. The next half of this function does exactly that.

```python
1         #Basically the accumulation of the most imperfect edges with the minimum
2         #length being defined by 180
3         #Thickness of the lines is 5
4         lines = cv2.HoughLinesP(croppedImg, 1, np.pi/180, 180, np.array([]), 180, 5)
5         #Now we need to find the slope, intercept and length of each of our detected lines
6         left_lines = []
7         length_left = []
8         right_lines = []
9         length_right = []
10        #We may not always detect a line that is why we do try/except statement
11        try:
12            for line in lines:
13                #Coordinates of a single line
14                for x1, y1, x2, y2 in line:
15                    #We dont want a vertical line or a horizontal line
16                    if x1==x2 or y1==y2:
17                        continue
18                    #Slope formula
19                    slope = (y2-y1)/(x2-x1)
20                    #Intercept
21                    intercept = y1 - slope*x1
22                    #Length
23                    length = np.sqrt((y2-y1)**2+(x2-x1)**2)
24                    #Y is reversed in images therefore a negative slope is a left line not right
25                    if slope<0:
26                        left_lines.append((slope, intercept))
27                        length_left.append((length))
28                    else:
29                        right_lines.append((slope, intercept))
30                        length_right.append((length))
31          #Now we have collected our similar lines into right and left lists
32          #Now we can convert them into lanes by dot product all the similar lines with lengths
33          #The longer lines are weighted more therefore affect the lanes more
34          #Then we normalise them by dividing by sum of the lengths(sort of like averaginng)
35          left_lane  = np.dot(length_left,  left_lines) /np.sum(length_left)  if len(length_left) >0 else None
36          right_lane = np.dot(length_right, right_lines)/np.sum(length_right) if len(length_right)>0 else None
37          #Now we have the right LANE and the left LANE through averaging and dot product
38          #Now we need to convert them back into coordinates for pixel points
39          #Having an equation of a line (assume infinite) we can select arbitrary points and find
40          #the x or y value accordingly.
41          #So we select arbitrary points for y1 = croppedImg.shape[0]
42          #and for y2 = y1*0.6, We need this in order to draw our lines (converting to pixel coordinates)
43          #We all need them to be int so cv2.line can use them
44          LeftX1 = int((croppedImg.shape[0] - left_lane[1])/left_lane[0])
45          LeftX2 = int(((croppedImg.shape[0]*0.6) - left_lane[1])/left_lane[0])
46          RightX1 = int((croppedImg.shape[0] - right_lane[1])/right_lane[0])
47          RightX2 = int(((croppedImg.shape[0]*0.6) - right_lane[1])/right_lane[0])
48          left_lane = ((LeftX1, int(croppedImg.shape[0])), (LeftX2, int(croppedImg.shape[0]*0.6)))
49          right_lane = ((RightX1, int(croppedImg.shape[0])), (RightX2, int(croppedImg.shape[0]*0.6)))
50          #Now we can draw them on the image
51          #We first create an empty array like our original image
52          #Then we draw the lines on the empty image and finally combine with our original image
53          emptImg = np.zeros_like(OrgImage)
54          #[255, 0, 0,]is the color, 20 is the thickness
55          #The star allows us to input a tuple (it processes as integer points)
56          cv2.line(emptImg, *left_lane, [255, 0, 0], 20)
57          cv2.line(emptImg, *right_lane, [255, 0, 0], 20)
58          #Finally we combine the two images
59          #It calculates the weighted sum of two arrays
60          #1.0 is the weight of our original image, we don't want to amplify it
61          #0.95 is the weight of our lines, and 0.0 is the scalar added to the sum
62          #be very significant in the image, just enough so we can see it and not obstruct anything else
63          finalImg = cv2.addWeighted(OrgImage, 1.0, emptImg, 0.95, 0.0)
64      except:
65          errors = True
66          print("Nothing detected")
67          #If we dont detect anything or to avoid errors we simply return the original image
68          return OrgImage, errors
69      #If all goes well, we return the image with the detected lanes
70      return finalImg, errors
```
**Line 4** The function HoughLines is quite difficult to understand. In laymans term it is simply detecting lines in our region and returning coordinates of those lines. We set a threshold of 180 essentially being the length and the thickness being 5. To find out more about hough transformation, go to this [link](http://www.swarthmore.edu/NatSci/mzucker1/opencv-2.4.10-docs/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html)

**Line 12-Line 30** Looping over all the detected lines we take one line at a time and we calculate the intercept and the slope. We omit horizontal and vertical lines as our lanes will never be straight in that perspective. Finally, depending on the slope we append our lines accordingly to the right and left lanes.

**Line 35-Line 36** We want to combine all our lines into lanes. We compute the dot product of the lines and their respective lengths. Longer lines have a heavier effect and so the slopes and intercepts of those line will be more dominant. Finally divide by the lengths to essentially normalise the values(So can be mapped to the image)

**Line 44-Line 47** We have the lanes, but in order to draw them we need coordinates. To draw any line (assuming infinite) you can pick arbitrary points, in this case I picked arbitrary y values. Using these I calculate the x values thus coordinates.

**Line 48-Line49** Now we just group those points accordingly to the right and left lanes.

**Line 53** We want to draw the line on top of our image. But in order to do that we need to have an overlay image. So here, we create an empty image with the same space dimensions as our original image. 

**Line 56-Line 57** Then we draw our lines on our empty image. The color used is blue as the format is BGR and not RGB.

**Line 63** Finally we combine the two images. This is done by calculating the weighted sum of the two arrays of images. In our empty image most of the pixels are set to 0 and so only the lane pixels will be effected in our original image.

**Line 65-Line 68** If there are any errors or a lane wasn't detected then we simply just output our original image.

**Line 70** If all goes well we output our final processed image.

Now we can go ahead and explore the next part of the code. The next part walks you through how we take the processed frames of our game and format it so we can get it ready to input it into our CNN.

```python
1     #Processes the images and returns the required data
2     def getFrames():
3         #We initialise the mss screenshot library
4         sct = mss.mss()
5         #This essentially takes a screenshot of the square from the coordinates
6         #You can adjust these to your liking, 
7         game = {'top': 122, 'left': 0, 'width': 512, 'height': 286}
8         #This converts the screenshot into a numpy array
9         gameImg = np.array(sct.grab(game))
10        #We want to resize the array so we can easily display it
11        gameImg = cv2.resize(gameImg, (600, 400))
12        #We pass the array into our calculateLanes function
13        #it returns our detected lanes image as well as if any errors were produced
14        img, errors = CalculateLanes(gameImg)
15        #You can show the render if you want with the lanes detections
16        cv2.imshow('window', img)
17        #To furhter process the image we convert it to a grayscale
18        img = cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_BGR2GRAY)
19        #In order for Keras to accept data we reshape it into the specific format
20        #I want to use an image thats 84x84
21        img = img.reshape(1, 84, 84)
22        #In order to give the algorithm the feel of the "velocity" we stack the 4 images
23        input_img = np.stack((img, img, img, img), axis = 3)
24        #This is required for openCV as a failsafe for stopping render
25        #By pressing q, you can stop render
26        if cv2.waitKey(25) & 0xFF == ord('q'):
27            cv2.destroyAllWindows()
28        #If all goes well we return the input_img and the errors
29        return input_img, errors
```
