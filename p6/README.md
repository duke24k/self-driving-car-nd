This project fuses the position and velocity measurements of obstacles from rader and lasar measurements to track the obstacles through time. It uses linear motion model. 

[//]: # (Image References)

[image1]: ./resources/kf_algorithm.jpg 
[image2]: ./resources/test.jpg 

## Kalman Filter Algorithm 

![alt text][image1]


## Dependencies

1. cmake >= 3.5
2. make >= 4.1
3. gcc/g++ >= 5.4


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./ExtendedKF


## Result
The Kalman Filter was able to track obstacles fairly accuractely with the sample measurements/ground truth that I used.

![alt text][image2]

