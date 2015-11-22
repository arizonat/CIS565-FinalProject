#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace ClearPath {
	struct agent{
		glm::vec3 pos;
		glm::vec3 vel;
		glm::vec3 goal;
		float radius;
		int id;
	};

	struct ray{
		glm::vec3 pos;
		glm::vec3 dir;
	};

	struct constraint{
		glm::vec3 norm;
		ray ray;
	};

	struct segment{
		glm::vec3 pos1;
		glm::vec3 pos2;
	};

	struct FVO{
		constraint R;
		constraint L;
		constraint T;
	};

	struct intersectionPoint{
		glm::vec3 point;
		bool isIntersection;
	};

	void initSimulation(int N);
	void stepSimulation(float dt);
	void copyAgentsToVBO(float *vbodptr, glm::vec2* endpoints, glm::vec3* pos, agent* agents);
}
