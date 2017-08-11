This project fuses the position and velocity measurements of obstacles from rader and lasar measurements to track the obstacles through time.  

[//]: # (Image References)

[image1]: ./resources/kf_algorithm.jpg 
[image2]: ./resources/test.jpg 
[image3]: ./resources/jacobian.jpg 



## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./ExtendedKF


## Accuracy

- RMSE is less than or equal to the values [.11, .11, 0.52, 0.52].

![alt text][image2]


## Follows the Correct Algorithm


- Kalman Filter Algorithm 

![alt text][image1]


I followed FusiionEKF implementation taught in Udacity class. 

- Creating an instance of the FusionEKF class.
- Receiving the measurement data calling the ProcessMeasurement() function. ProcessMeasurement() is responsible for the initialization of the Kalman filter as well as calling the prediction and update steps of the Kalman filter. ProcessMeasurement() function is implemented in FusionEKF.cpp.

## What is Kalman filter 

My UKF implementation is better than EKF implementation on Udacity simulator. Below is different Kalman filter explanation.

Thre is good discussion from udacity course forum about Kalman Filters. 

https://discussions.udacity.com/t/what-are-different-localization-methods/56140/2


Extended Kalman Filters - here the transition and measurement functions can be nonlinear, but you linearize them using a Taylor series and plug them into a regular Kalman filter. You make the same assumptions about the state transition and measurement noise.

![alt text][image3]


Unscented Kalman Filters - again the transition and measurement functions are nonlinear, but rather than a Taylor approximation, you use a different approximation scheme called the unscented transform and then feed the results into a Kalman filter.


Particle filters - here the transition and measurement functions are nonlinear. This is a sampling based approach unlike the other filters that are Kalman based. Essentially you approximate the posterior distribution with random samples and feed these directly through your transition and measurement functions.
