# face_recog_sal
This repository was made with the intention of educating Sal College students, since it would be utilized in a lecture for those students.
</br>This is simple to start
</br>Step 1 Create a Virtual environment in python using vevn module
Linux:
</br>
```sudo apt install python3-venv```
</br>Step 2:
After creating the venv check the version of pip and setup tool
</br>Use ```pip list``` command 
</br>Use command ```pip install module_name --upgrade``` to upgrade any module if needed
</br>
Let's Start the real thing!!
</br>Install the dependencies mentioned below 
</br>
```sudo apt-get install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev python3-dev python3-pip```
</br> Follow the below steps to install the dlib library
```
wget http://dlib.net/files/dlib-19.9.tar.bz2
tar xvf dlib-19.9.tar.bz2
cd dlib-19.9/
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd ..
pkg-config --libs --cflags dlib-1
```
After installing the dlib library install the other libraries using command
```pip install -r requirements.txt```
<br>
Create a ```train_dir``` folder in this folder create a sub folder with person's name and put at least 3 images of that person inside that folder 
```bash
----train_dir
    |
    |---Person_1
    |  |
    |  --images
    ---Person_2
       |
       --images
```
Run the code and you are ready to go
</br>
This is the source repo :- <a href="https://github.com/ageitgey/face_recognition">Face_recognition_github</a>
</br>
Also create a folder name ```unknown_capture```
