from datetime import datetime
import os
import face_recognition
import math
import cv2
import numpy as np

#Creating  a Dict to input all the users
user_list={}

#Function to read the log file 
def open_file(path):
    with open(path,'+r') as f :
        contents=f.readlines()
        temp_name=contents[-1]
        name=temp_name.split(' ')
        name_org=name[1].rstrip()
        user_list.update({name_org:temp_name})
        f.close()
#Fucntion to mark the PunchIN
def markAttendanceIN(name):
    open_file('AttendancePresent.csv')
    dt_string=datetime.now().strftime("%d/%m/%Y_%H:%M")
    
    temp_name=dt_string+" "+name+"\n"
    
    if (user_list=={}):
        write_to_file(temp_name)
        user_list.update({name:temp_name})
    else:
        if(name in user_list and user_list[name]== temp_name):
               return False
        else:
            write_to_file(temp_name)
            user_list.update({name:temp_name})

#Function for writing the logs in the file
def write_to_file(var):
    with open('AttendancePresent.csv', 'a+') as f:  
        f.write(var)
        f.close()
	
#Fucntion to mark the PunchOUT
def markAttendanceOUT(name):
    open_file('AttendanceAbsent.csv')
    dt_string=datetime.now().strftime("%d/%m/%Y_%H:%M")
    temp_name=dt_string+" "+name+"\n"
    if (user_list=={}):
        write_to_file(temp_name)
        user_list.update({name:temp_name})
    else:
        if(name in user_list and user_list[name]== temp_name):
               return False
        else:
            write_to_file(temp_name)  
            user_list.update({name:temp_name})

#Function to calculate the accuracy of the matched face
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0) 
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
#Function to process the video feed and compare the faces 
def processframe(ret,frame,known_face_encodings,classNames,Cam):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    small_frame=cv2.resize(frame,(0,0),fx=1,fy=1)
    rgb_frame = small_frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame,number_of_times_to_upsample=1,model="hog")   
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
   
    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = "Unknown" #Using the default name as unkonwn
        flag=False #creating a flag if the identified person is unknown
        if matches[best_match_index]:
            # print(face_distances)
	    # Calculating the accuracy
            acc=face_distance_to_conf(face_distances[best_match_index])
	#We only print the name if we are 90% sure that the face is matching one of many faces in our database
            if(acc>0.9):
		#Here based on the matched index we assign the name
                name = classNames[best_match_index] #if the person is identified then settign the name accordingly
                flag=True #setting the flag to true if the person is know 
        else:
            flag=False #else trying to reset the flag also checking the flag for each frame to avoid conflict
            
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        
	font = cv2.FONT_HERSHEY_DUPLEX
        
	if(flag):
            # For known
            percentage=round(acc,2) *100       
            f=name+"-"+str(percentage)
            cv2.putText(frame, f, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if (Cam==1): #checkign the flag for camera if cam is one then we mark punch in else punch out
                markAttendanceIN(name)
            else:
                markAttendanceOUT(name)

        else:
            #For unkown
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # marking attendence
            # markAttendanceOUT(name)
            if (Cam==1):
                if(name=="unknown"):
                    unkonwn_capture(frame,path='unknown_capture/')
                markAttendanceIN(name)
            else:
                if(name=="unknown"):
                    unkonwn_capture(frame,path='unknown_capture/')
                markAttendanceOUT(name)
           
# NOTE: commenting this to stop the thread conflict in raspberry pi
        frame_name='Video'+Cam
        cv2.imshow(frame_name, frame)  
	
#Fucntion to process the model learning 
def processlist(person,person_img,encodings,names):
        print(person,"Start")
        face_img=face_recognition.load_image_file("train_dir/"+person+"/"+person_img)
        #Getting the face Locations for each image
        face_loc=face_recognition.face_locations(face_img)
        #the above function will retrun an array/List of tuples
        #Now if a face a single face is found in a image then one we will train our else we will discard that image
        if len(face_loc)==1:
            #Providing face to the algo to train
            face_enc=face_recognition.face_encodings(face_img)[0]
            #The above function will learing from the image and will remember the encodings
            #adding this encodings to the encoding list
            encodings.append(face_enc)
            #adding the name of the person to the names list
            names.append(person)
        #Checking the condition for more than one face
        elif len(face_loc)>1:
            print(person + "/" + person_img + " was skipped => More than one face was detected")
        #IF not face is detected then we can go to this  
        else:
            print(person + "/" + person_img + " was skipped => No face was detected")
        print(person,"Done")
#Function to process the video stream.
def stream(video_capture,known_face_encodings,classNames,Cam):
    Camx=Cam
    while True:
        ret, frame = video_capture.read()
       
        processframe(ret,frame,known_face_encodings,classNames,Camx)

#         cv2.imshow('Video_cam', frame)
        # Hit 'q' on the keyboard to quit!c
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
	
#The below function is for capturing a person's image when he/she is unknown
def unkonwn_capture(frame,path):
    dir_list=os.listdir(path)
    now = datetime.now()
    capture_time=now.strftime("%d-%m-%Y %H:%M:%S")
    if dir_list==[]:
        print('Capturing 1st image')
        temp_path=capture_time+' Unknown_1.jpg'
        isWritten = cv2.imwrite(path+temp_path, frame)
    
    else:
        list_1=dir_list[-1].split('_')
        # print(list_1)
        list_counter=list_1[1].split('.')
        # print(list_counter)
        new_counter=int(list_counter[0])+1
        # ff.append('unkonwn_'+str(new_counter)+'.jpg')
        temp_path=capture_time+' Unknown_'+str(new_counter)+'.jpg'
        isWritten = cv2.imwrite(path+temp_path, frame)
    if isWritten:
	        print('Image is successfully saved as file.')
