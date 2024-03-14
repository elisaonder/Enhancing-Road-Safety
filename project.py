from __future__ import division
import numpy as np
import argparse
import cv2
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SeparableConv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D,LeakyReLU, PReLU
from keras.optimizers.legacy import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib as mp
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import threading
import pygame
from threading import Thread
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt 
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# command line argument 
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
a = ap.parse_args()
#change this do run the code in train or display mode
mode = 'display' 


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH) #landmark point
detector = dlib.get_frontal_face_detector() #extract face area

#alarm sound
sound = AudioSegment.from_wav('alarm.wav')


def plot_prova(history):
    #plot training and validation accuracy graph
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy']) 
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Plot training & validation loss graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape # Get the width and height of the input image
    if width is None and height is None:
        # If both width and height are None, return the original image unchanged
        return img
    elif width is None:
        # If only height is provided, calculate the ratio and resize accordingly
        ratio = height / h
        width = int(w * ratio) # Calculate the new width
        resized = cv2.resize(img, (height, width), interpolation) # Resize the image
        return resized
    else:
        # If the width is provided
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    # Initialize a numpy array to store the (x, y) coordinates of 68 facial landmarks
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36,48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
   
	# compute the eye aspect ratio
    ear = (A + B) / (1.5 * C)
 
	# return the eye aspect ratio
    return ear



def get_landmarks(im):
    # Detect faces in the input image using the 'detector' (dlib face detector)
    rects = detector(im, 1)
    
     # Check the number of detected faces
    if len(rects) > 1:
        # If more than one face is detected, return an error message
        return "error"
    if len(rects) == 0:
         # If no faces are detected, return an error message
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    # Create a copy of the input image to avoid modifying the original
    im = im.copy()
    for idx, point in enumerate(landmarks):
        # Extract the (x, y) coordinates of the landmark
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    # Initialize an empty list to store the points of the top lip
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    # Calculate the mean (average) point of the top lip along the y-axis
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    # Return the y-coordinate of the mean point as an integer
    return int(top_lip_mean[:,1])


def bottom_lip(landmarks):
    # Initialize an empty list to store the points of the bottom lip
    bottom_lip_pts = []
    # Extract points corresponding to the lower lip
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    # Calculate the mean (average) point of the bottom lip along the y-axis
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])



def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if "error" in landmarks:
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    # Calculate the lip distance (vertical distance between top and bottom lips)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


#variable for yawns calcuation initiation
yawns = 0
yawn_status = False 
output_text = " Yawn Count: 0"



# Define data generators
train_dir = '/Users/elisaonder/Desktop/ai-driver-safety-master/images/train'
val_dir = '/Users/elisaonder/Desktop/ai-driver-safety-master/images/validation'

#parameters of the model
num_train = 27709 # Number of training samples
num_val = 7178 # Number of validation samples
batch_size = 64 # Batch size for training
num_epoch = 50 # Number of epochs for training

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # Directory containing training images organized in subdirectories by class
        target_size=(48,48), # Resize images to the specified dimensions
        batch_size=batch_size,
        color_mode="grayscale", # Use grayscale images (single channel)
        class_mode='categorical') # Use categorical labels for multiclass classification

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(48, 48, 1)))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


if mode == "train":
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
    # Lists to store training and validation accuracy, and loss for each epoch
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    valid_loss = [] 

    # Loop through each epoch
    #compute training and validation loss and accuracy for each epoch
    for epoch in range(num_epoch):

        total_train_loss = 0.0
        total_train_accuracy = 0.0
        for step in range(num_train // batch_size):
            X_batch, y_batch = train_generator.next()
            loss, accuracy = model.train_on_batch(X_batch, y_batch)
            total_train_loss += loss
            total_train_accuracy += accuracy
            print('Training accuracy:'+ str(accuracy))
            print('Training loss:'+ str(loss))
        
        # Average training accuracy and loss for the epoch
        train_accuracy.append(total_train_accuracy/(num_train // batch_size))
        train_loss.append(total_train_loss/(num_train // batch_size))

        total_val_accuracy = 0.0
        for step in range(num_val // batch_size):
            X_val_batch, y_val_batch = validation_generator.next()
            val_loss, val_accuracy_batch = model.test_on_batch(X_val_batch, y_val_batch)
            total_val_accuracy += val_accuracy_batch
            print('Validation accuracy:'+ str(val_accuracy_batch))
            print('Validation loss:'+ str(val_loss))
    
        # Average validation accuracy and loss for the epoch
        val_accuracy.append(total_val_accuracy / (num_val // batch_size))
        valid_loss.append(total_val_accuracy / (num_val // batch_size))

    model.save_weights('model.h5')

    #plot training and validation accuracy graph
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    #plot training and validation loss graph
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

    """
    print(model_info.history)
    plot_prova(model_info)
    model.save_weights('model.h5')
    model.summary()

    import visualkeras
    visualkeras.layered_view(model).show() # display using your system viewer
    visualkeras.layered_view(model, to_file='output.png') # write to disk
    visualkeras.layered_view(model, to_file='output.png').show() # write and show

    visualkeras.layered_view(model)


    import keras_sequential_ascii
    from keras_sequential_ascii import keras2ascii
    keras2ascii(model)
    """


#this part runs only if we are not in the training mode
elif mode == "display":
    #load the model 
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam
    cap = cv2.VideoCapture(0)


    predictor_path = 'shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    total=0
    alarm=True
    det_face="False"


    ear_plot=[]
    yawn_plot=[]
    true_labels = []
    predicted_labels = []

    while True:
        
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            break
        facecasc = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        frame_resized = resize(gray, width=120)
        cv2.putText(frame, "Number of seconds the eyes are closed:", (10, 30),cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
        
# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
        dets = detector(frame_resized, 1)

        if len(dets) > 0:
            for k, d in enumerate(dets):
                shape = predictor(frame_resized, d)
                shape = shape_to_np(shape)
                leftEye= shape[lStart:lEnd]
                rightEye= shape[rStart:rEnd]
                leftEAR= eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                #eye blinking is perform by both eyes 
                #here we do the average
                ear = (leftEAR + rightEAR) / 2.0
                
                ear_plot.append(ear)


                if ear>.25: #eyes not closed
                    print (round(ear,2))
                    total=0
                    alarm=False
                    #cv2.putText(frame, "Number of seconds the eyes are closed:", (10, 30),cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
                else: #if eyes are closed
                    total+=1
                    if total>3:
                        total=0
                        cv2.putText(frame, "drowsiness detected" ,(150, 400),cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 4)
                              
                        if not alarm:
                            
                            alarm=True
                            # check to see if an alarm file was supplied,
                            # play the alarm until press a
                              
                            while cv2.waitKey(1) & 0xFF != ord('a'):  #to quit the program press a
                        
                                play(sound)
                            
                    #cv2.putText(frame, "Number of seconds the eyes are closed:".format(total), (10, 30),cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
                   
                        
                #eyes landmarks
                for (x, y) in shape:
                   cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
        
        #put text on the video
        cv2.putText(frame, str(total), (730, 30),cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
        #cv2.imshow("image", frame)

        image_landmarks, lip_distance = mouth_open(frame)
        
        yawn_plot.append(lip_distance)

        prev_yawn_status = yawn_status  
        
        if lip_distance > 60: #if yawn
            yawn_status = True 
            
            cv2.putText(frame, "Subject is Yawning", (10, 145), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1,(0,0,255),2)
            

            output_text = " Yawn Count: " + str(yawns + 1)
            if yawns==1: #first yawns within 60 seconds
                inizio = time.time()
           
            if yawns>=5 and inizio- time.time()<=60: #more then 4 yawns within 60 seconds
                
                #start the alarm until press a
                while cv2.waitKey(1) & 0xFF != ord('a'):  #to quit the program press a
                    play(sound)
                
                #reset number of yawns
                yawns=0
                #put the text on the video
                cv2.putText(frame, "drowsiness detected" ,(150, 400),cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 4)
            

            cv2.putText(frame, output_text, (0,70),
                        cv2.FONT_HERSHEY_TRIPLEX, 1,(0,255,0),2)
        else:
            cv2.putText(frame, output_text, (0,70),
                        cv2.FONT_HERSHEY_TRIPLEX, 1,(0,255,0),2)
            yawn_status = False 
         
        if prev_yawn_status == True and yawn_status == False:
            yawns += 1
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 255), 2)
            det_face="True"
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
           
            #perccentage also depends on the emotion status of the driving person
            p=0
            if emotion_dict[maxindex]=="Angry":
                em=40
            elif emotion_dict[maxindex]=="Surprised":
                em=30
            elif emotion_dict[maxindex]=="Fearful":
                em=20
            elif emotion_dict[maxindex]=="Sad":
                em=15
            else:
                em=0

            #calculation of the total percentage
            p = yawns * 3 + total * 15 + em

            # Get the true label (ground truth) from the emotion dictionary
            true_label = [k for k, v in emotion_dict.items() if v == emotion_dict[maxindex]][0]

            # Append the true and predicted labels to the lists
            true_labels.append(true_label)
            predicted_labels.append(maxindex)

            cv2.putText(frame, "Accident Risk: "+str(p)+"%", (10, 110), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        
        #if face not detected so person is not looking at the road
        if det_face=="False":
            st = time.time()
            det_face="False2"

        if det_face=="True":
            det_face="False"

        

        #if person is not looking at the road for more then or equal to 3 seconds
        #start the alarm
        if det_face=="False2" and time.time()-st>=3:
            
            #start the alarm 
            #press a to stop it
            while cv2.waitKey(1) & 0xFF != ord('a'):  #to quit the program press a
                play(sound)
            
            cv2.putText(frame, "drowsiness detected" ,(150, 400),cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 4)
            det_face="True"

            #cv2.putText(frame, "hhhhhhhhhh", (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
           
        
       

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  #to quit the program press q
           break 


    '''
    #Graph ear value vs time
    
    plt.plot(ear_plot)
    plt.title('EAR over time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('EAR')
    plt.show()
    
    #plot lips distance rate over time
    
    plt.plot(yawn_plot)
    plt.title('Lips distance over time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Lips distance')
    plt.show()
    '''
   
    #cv2.imshow('Live Landmarks', image_landmarks )
    #cv2.imshow('Yawn Detection', frame )
    
  