import cv2
from sklearn import svm
import os
from  util  import *
from datetime import datetime

#Creating the lists to store encodings,names and classnames
encodings=[]
names=[]
className=[]

#Fetching the data to train the model from train_dir
train_dir=os.listdir('train_dir/')

thread_pool=[]
#Loop for fetching the folder for each image
for person in train_dir:
    #print("Training model for : "+person)
    #creating a list to store all this folders
    pix=os.listdir("train_dir/"+person+"/")
    
    #loop to go thorough each folder and fetch images
    for person_img in pix:
        #loading the images to the algo
        # thread_pool.append(threading.Thread(target=processlist,args=(person,person_img,encodings,names)))
        processlist(person,person_img,encodings,names)

# for threads in thread_pool:
#     threads.start()
#     threads.join()


# for threads in thread_pool:

#Creating and training the SVC classifier
clf =svm.SVC(gamma='scale')
clf.fit(encodings,names)
print("Training Complete")
    
known_face_encodings = encodings
classNames=names
print('Encoding Complete')

video_capture = cv2.VideoCapture("http://192.168.55.37:81/stream")
# video_capture = cv2.VideoCapture(0)

count=0

while True:
    dt=datetime.now()
    # print("Fetching image ",dt)
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # dtx=datetime.now()

    # print("done fetching image ",dtx)
    # if(count%3==0):
        # threading.Thread(target=processframe,args=(ret,frame,known_face_encodings,classNames)).start()
    processframe(ret,frame,known_face_encodings,classNames,1)

    cv2.imshow('Video_cam', frame)
    # print(count)
    # count=count+1
    # if(count>=10):
    #     count=0
    # Hit 'q' on the keyboard to quit!c
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()