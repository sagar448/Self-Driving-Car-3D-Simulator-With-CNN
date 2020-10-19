# =============================================================================
# Written by Sagar Jaiswal
# =============================================================================
import cv2
import mss
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
import pyautogui as p
import random
import time
GPU = 1 #Change it to 0 in order to use CPU

if GPU == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
#Function calculates the lanes
def CalculateLanes(OrgImage):
    errors = False
    #Since our game has yellow lanes, we can detect a specific color
    #keep that color, and get rid of everything else to make it easier
    #to detect the yellow lanes
    #So we convert our image to the HLS color scheme
    HLSImg = cv2.cvtColor(OrgImage, cv2.COLOR_BGR2HLS)
    #The lower and upper arrays define boundaries of the BGR color space
    #BGR because OpenCV represents images in Numpy in reverse order
    #So for our yellow color we say that our pixels color that are yellow will be
    # R>= 100, B >= 0, G>=10 (lower limit), R<=255, B<=255, G<=40
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    #inRange basically finds the color we want in the HLSImg with the lower and upper
    #boundaries(the ranges)
    yellow_mask = cv2.inRange(HLSImg, lower, upper)
    #We then apply this mask to our original image, and this returns an image showing
    #only the pixels that fall in the range of that mask
    YellowImg = cv2.bitwise_and(OrgImage, OrgImage, mask=yellow_mask)
    #Convert the original image to gray
    GrayImg = cv2.cvtColor(YellowImg, cv2.COLOR_BGR2GRAY)
    #Apply blurring
    #The 5x5 is the gaussianblur kernel convolved with image
    #The 0 is the sigmaX and SigmaY standard deviation usually taken as 0
    blurredImg = cv2.GaussianBlur(GrayImg, (5, 5), 0)
    #Detect edges in the image
    #700 is the max val, any edges above the intensity gradient of 700 are edges
    #200 is the lowest intensity gradient, anything below is not an edge
    imageWithEdges = cv2.Canny(blurredImg, threshold1=200, threshold2=700)
    #These are the points of our trapezoid/hexagon that we crop out 
    points = np.array([[0, 310],[0, 300], [220, 210], [380, 210], [600, 300], [600, 310]])
    #Now we calculate the region of interest
    #We first create a mask (a blank black array same size as our image)
    mask = np.zeros_like(imageWithEdges)
    #Then we fill the mask underneath the points(inside the polygon) we defined
    #with color white (255)(This is the part of the image we want to process)
    cv2.fillPoly(mask, [points], 255)
    #this bitwise and function basically combines the two images
    #The coloured bit where the pixels had a value 255 is kept while the
    #top bit is removed (which is 0)
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
        #Now we have collected our similar lines into right and left lists
        #Now we can convert them into lanes by dot product all the similar lines with lengths
        #The longer lines are weighted more therefore affect the lanes more
        #Then we normalise them by dividing by sum of the lengths(sort of like averaginng)
        left_lane  = np.dot(length_left,  left_lines) /np.sum(length_left)  if len(length_left) >0 else None
        right_lane = np.dot(length_right, right_lines)/np.sum(length_right) if len(length_right)>0 else None
        #Now we have the right LANE and the left LANE through averaging and dot product
        #Now we need to convert them back into coordinates for pixel points
        #Having an equation of a line (assume infinite) we can select arbitrary points and find
        #the x or y value accordingly.
        #So we select arbitrary points for y1 = croppedImg.shape[0]
        #and for y2 = y1*0.6
        #We all need them to be int so cv2.line can use them
        LeftX1 = int((croppedImg.shape[0] - left_lane[1])/left_lane[0])
        LeftX2 = int(((croppedImg.shape[0]*0.6) - left_lane[1])/left_lane[0])
        RightX1 = int((croppedImg.shape[0] - right_lane[1])/right_lane[0])
        RightX2 = int(((croppedImg.shape[0]*0.6) - right_lane[1])/right_lane[0])
        left_lane = ((LeftX1, int(croppedImg.shape[0])), (LeftX2, int(croppedImg.shape[0]*0.6)))
        right_lane = ((RightX1, int(croppedImg.shape[0])), (RightX2, int(croppedImg.shape[0]*0.6)))
        #Now we can draw them on the image
        #We first create an empty array like our original image
        #Then we draw the lines on the empty image and finally combine with our original image
        emptImg = np.zeros_like(OrgImage)
        #[255, 0, 0,]is the color, 20 is the thickness
        #The star allows us to input a tuple (it processes as integer points)
        cv2.line(emptImg, *left_lane, [255, 0, 0], 20)
        cv2.line(emptImg, *right_lane, [255, 0, ], 20)
        #Finally we combine the two images
        #It calculates the weighted sum of two arrays
        #1.0 is the weight of our original image, we don't want to amplify it
        #0.95 is the weight of our lines, we don't set it to 1 because we don't want it to
        #be very significant in the image, just enough so we can see it and not obstruct anything else
        finalImg = cv2.addWeighted(OrgImage, 1.0, emptImg, 0.95, 0.0)
    except:
        errors = True
        print("Nothing detected")
        #If we dont detect anything or to avoid errors we simply return the original image
        return OrgImage, errors
    #If all goes well, we return the image with the detected lanes
    return finalImg, errors

#Processes the images and returns the required data
def getFrames():
    #We initialise the mss screenshot library
    sct = mss.mss()
    #This essentially takes a screenshot of the square from the coordinates
    #You can adjust these to your liking, 
    game = {'top': 122, 'left': 0, 'width': 512, 'height': 286}
    #This converts the screenshot into a numpy array
    gameImg = np.array(sct.grab(game))
    #We want to resize the array so we can easily display it
    gameImg = cv2.resize(gameImg, (600, 400))
    #We pass the array into our calculateLanes function
    #it returns our detected lanes image as well as if any errors were produced
    img, errors = CalculateLanes(gameImg)
    #You can show the render if you want with the lanes detections
    cv2.imshow('window', img)
    #To further process the image we convert it to a grayscale
    img = cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_BGR2GRAY)
    #In order for Keras to accept data we reshape it into the specific format
    #I want to use an image thats 84x84
    img = img.reshape(1, 84, 84)
    #In order to give the algorithm the feel of the "velocity" we stack the 4 images
    input_img = np.stack((img, img, img, img), axis = 3)
    #This is required for openCV as a failsafe for stopping render
    #By pressing q, you can stop render
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    #If all goes well we return the input_img and the errors
    return input_img, errors

#This function makes the car accelerate
def straight():
    p.keyDown("up")
    p.keyUp("right")
    p.keyUp("left")
    p.keyUp("up")

#We can turn right with this
def right():
    p.keyDown("right")
    p.keyUp("right")

#Turn left with this
def left():
    p.keyDown("left")
    p.keyUp("left")


#For now we make the car accelerate, turn right and turn left
moves = 3
#learning rate (discount rate)
learningRate = 0.9
#This is the exploration rate (epsilon)
#Its better at first to let the model try everything
epsilon = 1.0
#We don't want our model to stop exploring so we set a minimum epsilon
epsilon_min = 0.01
#We also dont want our model to explore all the time therefore we want it
#to decay
epsilon_decay = 0.995
#Number of times we want to train the algorithm
epochs = 100
#We want to store our data for replay/so our model can remember
memory = []
#The max amount of stuff we want to remember
max_memory = 500

#Lets start defining our model
model = Sequential()
#We will be using a CNN with 32 filters, 3x3 kernel and the input shape will be
#84x84 with 4 grayscale images stacked on top
#padding will be set as same and we will use the rectified activation function
model.add(Conv2D(32, (3, 3), input_shape=(84, 84, 4), padding='same',
                 activation='relu'))
#This time we will use 64 filters with a 3x3 kernel, with the same act function 
#but the padding will change
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
#We flatten our data in order to feed it through the dense(output) layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
#We have 3 outputs, forward, left, right
model.add(Dense(3, activation='linear'))
#We will be using the mean squared error
model.compile(loss='mean_squared_error',
              optimizer=SGD())
#loop over the number of epochs (essentially the number of games)
for i in range(epochs):
    time.sleep(5)
    #We set the game_over to false as the game is just starting
    game_over = False
    #We start of by getting initial frames and errors
    input_img, errors = getFrames()
    #We set the errors to false to begin with
    errors = False
    #We set the reward to 0
    reward = 0
    #While the game is not over we loop
    while game_over==False:
        #Np.random.rand() returns a number between 0 and 1
        #We check if its smaller that our exploration factor
        if np.random.rand() <= epsilon:
            #if the random number is smaller than our exploration factor
            #We select a random action from our 3 actions
            action = np.random.randint(0, moves, size=1)[0]
        else:
            #If it's not smaller than we predict an output by inputting our
            #4 stacked images
            #ouput is the probability of our 3 directions
            output = model.predict(input_img)
            #action is the index of the highest probability and therefore
            #indicates which turn to take
            action = np.argmax(output[0])
        #if our action == 0 then we go straight   
        if int(action) == 0:
            straight()
        #If our action == 1 then we go right
        elif int(action) == 1:
            right()
        #else we go left
        else:
            left()
        #Once we've performed our action we get the next frame
        #We also check weather to reward the algorithm or not
        input_next_img, errors = getFrames()
        #If we detect lanes and therefore no errors occur we reward the algorithm
        if errors == False:
            reward = reward + 1
        #Else if there we detect no lanes and so there is an error we 
        #say its game over
        else:
            game_over = True
        #Game over or not we want to keep record of the steps the algo took
        #We first check if the total memoery length is bigger than the max memory
        if len(memory) >= max_memory:
            #If more memory then needed we delete the first ever element we added
            del memory[0]
        #We append it to our memory list
        memory.append((input_img, action, reward, input_next_img, game_over))
        #Next we set our input_img to our latest data
        input_img = input_next_img
        if game_over:
            print("Game: {}/{}, Total Reward: {}".format(i, epochs, reward))
    #Once the game is over we want to train our algo with the data we just collected
    #We check if our memory length is bigger than our batch size 
    if len(memory) > 32:
    #If so then we set the batch_size to 32
        batch_size = 32
    else:
    #Else we set our batch size to whatever is in the memory
        batch_size = len(memory)
    #We are taking a random sample of 32 so not to overfit our algo
    batch = random.sample(memory, batch_size)
    #We itereate over every memory we've stored in that memory batch of 32
    for input_img, action, reward, input_next_img, game_over in batch:
        #if in that memory our game was over then we set the target_reward equal to reward
        target_reward = reward
        #If our game was not over
        if game_over == False:
        #This essentially is the bellman equation
        #expected long-term reward for a given action is equal to the 
        #immediate reward from the current action combined with the expected 
        #reward from the best future action taken at the following state.
        #The model isn't certain that for that specific action it will get the best reward
        #It's based on probability of the action, if the probability of that action is in the
        #negatives then our future reward is going to be further decreased by our learning rate
        #This is just the model being cautious, as to not set an impossible reward target
        #If the reward is impossible then the algorithm might not converge
        #Converge as in a stable condition where it can play the game without messing up
            target_reward = reward + learningRate * \
            np.amax(model.predict(input_next_img)[0])
        #So from above we essentially know what is going to happen(input_next_img) 
        #assuming the game wasn't over, the algorithm did well.
        #So we want the algorithm to perform the same, essentially we
        #persuade the algorithm to do what it did to get that reward
        #so we make the algorithm predict from the previous frame(input_img)
        #but we alter its prediction according to the action that got the highest
        #reward and...
        desired_target = model.predict(input_img)
        #we set that as the target_reward...
        desired_target[0][action] = target_reward
        #So to make the algo perform the same, we associate the input_img with the
        #target we want and we fit it
        model.fit(input_img, desired_target, epochs=1, verbose=0)
    #Finally we check if our exploration factor is bigger than our minimum exploration
    #if so we decrease it by the decay to reduce exploration, we do this every game
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
