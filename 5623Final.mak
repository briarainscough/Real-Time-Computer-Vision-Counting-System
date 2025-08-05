Hello! This is my 5623 Final project makefile!

Please compile the code with

 	g++ syslog_rt.cpp -o sysrt `pkg-config --cflags --libs opencv4` -pthread -std=c++20

and run it with 
	
	sudo ./sysrt

To access the syslog, grep.

To get the gesture recognition to work, must wear a black glove. 
