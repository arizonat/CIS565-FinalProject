#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace ClearPath {

	struct HRVO{
		glm::vec3 apex;
		glm::vec3 left;
		glm::vec3 right;
	};

	struct CandidateVel{
		glm::vec3 vel;
		bool valid;
		float distToPref;
		int hrvo1;
		int hrvo2;
	};

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

	struct UGComp{
		__host__ __device__ bool operator()(const UGEntry &a, const UGEntry &b){
			return a.cellId < b.cellId;
		}
	};

	void initSimulation(int N);
	void stepSimulation(float dt, int iter);
	void copyAgentsToVBO(float *vbodptr, glm::vec3* pos, agent* agents, HRVO* hrvos, CandidateVel* candidates, int* neighbors, int* num_neighbors);
}
