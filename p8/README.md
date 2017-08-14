This project fuses the position, velocity & yaw_rate measurements of particles from udacity simulator to localize the position of car through time.  

[//]: # (Image References)

[image0]: ./resources/test.jpg 
[image1]: ./resources/pf_equation.jpg 


## Basic Build Instructions

This repo contains 10 projects and too large to clone. Alternatively, p8.zip file can be downloaded. (p8.zip contains the project file.)

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./particle_filter


## Accuracy

The particle filter was able to pass the track.

![alt text][image0]

## Follows the Correct Algorithm

The transition and measurement functions are nonlinear. This is a sampling based approach unlike the other filters that are Kalman based. Essentially you approximate the posterior distribution with random samples and feed these directly through your transition and measurement functions.


In that point, PF implementation is different from EKF and UKF implementation. 

 ![alt text][image1]

 ## Discussion 

I used udacity simulator to implement particle filter that provides simulated data of sensors. 
Below code in main.cpp is provided by Udacity. 



    if (!pf.initialized()) {
      
      double sense_x = std::stod(j[1]["sense_x"].get<std::string>());
	  double sense_y = std::stod(j[1]["sense_y"].get<std::string>());
	  double sense_theta = std::stod(j[1]["sense_theta"].get<std::string>());
     
	  pf.init(sense_x, sense_y, sense_theta, sigma_pos);

		}

	else {

		  double previous_velocity = std::stod(j[1]["previous_velocity"].get<std::string>());
		  double previous_yawrate = std::stod(j[1]["previous_yawrate"].get<std::string>());

		  pf.prediction(delta_t, sigma_pos, previous_velocity, previous_yawrate);
		}



Raw data processing that get those data will be also necessary to implement particle filter.  

