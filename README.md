# Real-Time-Computer-Vision-Counting-System
Uses real time scheduling and computer vision to interpret the number of fingers present in front of a camera. 

Designed to run on RPi3 and Logitech UVC Camera.  

Please compile the code with:

 	g++ syslog_rt.cpp -o sysrt `pkg-config --cflags --libs opencv4` -pthread -std=c++20

and run it with:
	
	sudo ./sysrt

To access the syslog, grep.

To get the gesture recognition to work, must wear a black glove. Works best on white background. 
