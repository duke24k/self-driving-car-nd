
[//]: # (Image References)


## Basic Build Instructions

This repo contains 10 projects and too large to clone. Alternatively, p9.zip file can be downloaded. (p9.zip contains the project file.)

1. Clone this repo.
2. Make a build directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./pid


## Result

The PID controller was able to pass the track. 
test.mp4 file contains the result.

## Tunning parameters of PID.

Taking a snippet from Wikipedia
(https://en.wikipedia.org/wiki/PID_controller) 

CTE is provided by the simulator at every iteration. Therefore, I just used below equation for cte without dt.

```
void PID::UpdateError(double cte) {

  sum_cte += cte;
    p_error = - Kp * cte;
    i_error = - Ki * sum_cte;
    d_error = - Kd * (cte - prev_cte);
    prev_cte = cte;

}
```

I manually tunned each component. Kp and Kd are tuned for the car to minimize oscillation on the straight and curved lane. Ki is tuned based on CTE value when the car drives on the straight lane, after I tuned Kp and Kd.

