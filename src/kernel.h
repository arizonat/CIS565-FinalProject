#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace ClearPath {

	struct Cell{
		int x;
		int y;
	};

	struct UGEntry{
		int cellId;
		int agentId;
	};

	struct agent{
		glm::vec3 pos;
		glm::vec3 vel;
		glm::vec3 goal;
		float radius;
		int id;
		float w;
	};

	struct ray{
		glm::vec3 pos;
		glm::vec3 dir;
	};

	struct constraint{
		glm::vec3 norm;
		ray ray;
		glm::vec3 endpoint;
		bool isRay;
	};

	struct segment{
		glm::vec3 pos1;
		glm::vec3 pos2;
		bool isOutside;
	};

	struct FVO{
		constraint R;
		constraint L;
		constraint T;
	};

	struct intersection{
		glm::vec3 point;
		bool isOutside;
		bool isIntersection;
		float distToOrigin; // Distance to the constraint origin
		//float distToRayOrigin;
	};

	struct UGComp{
		__host__ __device__ bool operator()(const UGEntry &a, const UGEntry &b){
			return a.cellId < b.cellId;
		}
	};

	struct DistToOriginComp{
		__host__ __device__ bool operator()(const intersection &a, const intersection &b){
			return a.distToOrigin < b.distToOrigin;
		}
	};

	void initSimulation(int N);
	void stepSimulation(float dt, int iter);
	void copyAgentsToVBO(float *vbodptr, glm::vec2* endpoints, glm::vec3* pos, agent* agents, intersection* intersections, int* neighbors, int* num_neighbors);
}
