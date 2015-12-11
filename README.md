# CIS565 Fall 2015 - Final Project

## CUDA-accelerated Crowd Simulation with HRVO and ClearPath

Levi Cai

![](img/200_robots.PNG)

Final video: https://www.youtube.com/watch?v=9E0RYBbbvJQ&feature=youtu.be

Final presentation: https://docs.google.com/presentation/d/1MXH-gVVsqWDP741cm3D0KGq5vxWAhtXCxeXZsPsarFs/edit?usp=sharing

Papers/Resources Used:
* NOTE: Currently used intersection implementation from Snape's HRVO implementation: https://github.com/snape/HRVO
* ClearPath paper: http://gamma.cs.unc.edu/CA/ClearPath.pdf
* General collision avoidance: http://gamma.cs.unc.edu/CA/
* RVO paper: http://gamma.cs.unc.edu/RVO/
* HRVO: http://gamma.cs.unc.edu/HRVO/
* HRVO paper: http://gamma.cs.unc.edu/HRVO/HRVO-T-RO.pdf

Special thanks to the authors: Prof. Stephen Guy and Prof. Ming Lin

## Debug view:

![](img/debug_view_labelled.png)

## Final Algorithm Overview

![](img/hrvo_30.PNG)

![](img/hrvo_38_random.PNG)

### General Approach

The general approach to all of the above algorithms for collision avoidance is to model all velocities of every agent that *will* results in a collision. Then, if the current velocity is inside the set of colliding-velocities, to find the velocity nearest to the original, but is no longer in the set of colliding velocities.

### Optimizations

I implemented a uniform grid for nearest neighbor computations as well as a few implementations of stream compaction for the parallelization steps in order to get semi-reasonable performance. There are some other ideas present in the ClearPath paper for additional optimizations, but I did not have a chance to implement them fully. It is important to note that the Uniform Grid actually did not improve performance that much for smaller samples of robots (< 200), which is the only set that I have stable simulations for.

## Things I tried

#### FVO with ClearPath

![](img/fvo_3.PNG)

For my first attempt, I implemented the algorithm from the **ClearPath paper**, including FVOs (3 constraints), and the full intersection computation and inside/outside classifications. The full code for this implementation is available at commit aab96587f677be6439d2f293825ed827a7786f8d.

The idea for this algorithm is to model 3 different contraints. These 3 linear constraints are the boundaries of the velocities that result in collisions. Once all the agent pair-wise constraints have been computed, we then compute the intersection of all the boundaries, label them as inside/outside the colliding region of the other pair-wise constraints, and then find the velocity along the segments that are outside the other constraints. This velocity, if computed properly, is thus guaranteed to avoid collisions. However, due to the 3rd constraint, there are lots of symmetries and small epsilon differences that greatly effect the resulting inside/outside classifications and thus can give strange results.

As clarified by Prof. Guy, is highly susceptible to floating point issues when using the 3rd constraint and the intersection classification algorithm. But I did get it to work for 3 robots: https://www.youtube.com/watch?v=VDhIFXhSx-o
and here it is breaking on 4: https://www.youtube.com/watch?v=BA9yAxEHa7g

#### RVO with randomly sampled velocities

![](img/rvo_6.PNG)

For my second attempt, I implemented the algorithm from the **RVO paper**. Here the difference is that we no longer compute intersection points and find velocities along the boundaries. Instead we compute the velcoity regions that result in collisions as before, but for each agent we randomly sample _N_ velocities (I used _N_ = 250 as in the paper). We then select the velocity that is nearest to the desired velocity AND is not inside the colliding regions. This resulted in a slightly easier-to-compute algorithm and worked fairly well for up to 6 robots: https://www.youtube.com/watch?v=HOVsiz8CZ-4&feature=youtu.be. Beyond that it seemed to give strange results.

#### HRVO with ClearPath

Then from Prof. Stephen Guy's advice, I implemented HRVO with ClearPath on CUDA. See final algorithm above for details.

## Performance Analysis

Note: This is not heavily optimized and it not a particularly stable implementation, so there could be differences once things are fixed more.

![](img/gpu_vs_cpu.png)

Units on left are ms.

![](img/sparse_uniform_grid_comparison.png)

![](img/dense_uniform_grid_comparison.png)

![](img/sparse_kernel.png)

![](img/dense_kernel.png)

