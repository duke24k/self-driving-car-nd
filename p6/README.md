This project fuses the position and velocity measurements of obstacles from rader and lasar measurements to track the obstacles through time. It uses linear motion model. 

[//]: # (Image References)

[image1]: ./resources/kf_algorithm.jpg 
[image2]: ./resources/test.jpg 

---
Kalman Filter Algorithm 

![alt text][image1]

---
##Dependencies

cmake >= 3.5
All OSes: click here for installation instructions
make >= 4.1
Linux: make is installed by default on most Linux distros
Mac: install Xcode command line tools to get make
Windows: Click here for installation instructions
gcc/g++ >= 5.4
Linux: gcc / g++ is installed by default on most Linux distros
Mac: same deal as make - install Xcode command line tools
Windows: recommend using MinGW

---
##Basic Build Instructions

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
   On windows, you may need to run: cmake .. -G "Unix Makefiles" && make
4. Run it: ./ExtendedKF

---
##Result
The Kalman Filter was able to track obstacles fairly accuractely with the sample measurements/ground truth that I used.

![alt text][image2]

