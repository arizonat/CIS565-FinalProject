CIS565 Fall 2015 - Final Project

CUDA-accelerated Crowd Simulation with HRVO and ClearPath

Levi Cai

NOTE: Currently used intersection implementation from Snape's HRVO implementation: https://github.com/snape/HRVO

![](img/200_robots.PNG)

Papers/Resources Used:

* ClearPath algorithm: http://gamma.cs.unc.edu/CA/ClearPath.pdf
* General collision avoidance: http://gamma.cs.unc.edu/CA/
* RVO with random sampling: http://gamma.cs.unc.edu/RVO/
* HRVO: http://gamma.cs.unc.edu/HRVO/
* HRVO paper: http://gamma.cs.unc.edu/HRVO/HRVO-T-RO.pdf


The paper I will be referencing can be found here: http://gamma.cs.unc.edu/CA/ClearPath.pdf


Additional videos and materials can be found here: http://gamma.cs.unc.edu/CA/


Initial Demo


Hope to have a simple version of the P-ClearPath algorithm running with the simple N-body simulator, probably just in 2D since all the P-ClearPath examples are in 2D. No optimizations expected, just want to see at least 2-5 robots avoid each other.


Milestone 1

Implement CPU version for comparison? 

Milestone 2

Implement the Uniform Grid optimization

Milestone 3

ALLLLL the performance checks. If there is time, extend it for non-straight paths in environments with obstacles.
