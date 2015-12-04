#define GLM_FORCE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/constants.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
//#include <vector>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define sign(x) (x>0)-(x<0)

//#define USE_FVO 0
//#define USE_SAMPLING 0
#define USE_HRVO 1

//#define INFINITY 0x7f800000
#define NEG_INFINITY 0xff800000

#define MAX_NEIGHBORS 10

using namespace ClearPath;

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAError(const char *msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128
#define robot_radius 0.5
#define circle_radius 10
#define desired_speed 3.0f

#define GRIDMAX 20 // Must be even, creates grid of GRIDMAX x GRIDMAX size
#define NNRADIUS 2.0f

// Experimental sampling
#define MAX_VEL 3.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numAgents;
// TODO: bottom few will need to get removed (# will vary in the loop)

int numNeighbors; // Neighbors per agent
int totNeighbors;

int numHRVOs;
int totHRVOs;
int numCandidates;
int totCandidates;

dim3 threadsPerBlock(blockSize);

float scene_scale = 1.0;

glm::vec3* dev_pos;
agent* dev_agents;
int* dev_neighbors; //index of neighbors
bool* dev_in_pcr;
UGEntry* dev_uglist;
int* dev_startIdx;

glm::vec3* dev_vel_new;

int* dev_ug_neighbors;
int* dev_num_neighbors;

bool* dev_indicators;
int* dev_indicator_sum;
int* dev_idx;

int* dev_agent_ids;
int* dev_neighbor_ids;

HRVO* dev_hrvos;
CandidateVel* dev_candidates;

/*******************
* Uniform Grid Functions *
********************/

__host__ __device__ Cell getGridCell(float radius, agent a){
	Cell cell;

	// Grid top-left corner
	glm::vec3 tlc = glm::vec3(-GRIDMAX/2,GRIDMAX/2,0) * radius;
	glm::vec3 dist = glm::abs(a.pos - tlc);
	cell.x = int(dist.x / radius);
	cell.y = int(dist.y / radius);

	return cell;
}

__host__ __device__ int getIdFromCell(Cell cell){
	return cell.x + cell.y*GRIDMAX;
}

__host__ __device__ Cell getCellFromId(int id){
	Cell cell;
	cell.x = id % GRIDMAX;
	cell.y = id / GRIDMAX;
	return cell;
}

__global__ void kernUpdateUniformGrid(int numAgents, UGEntry* uglist, agent* agents){
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < numAgents){
		UGEntry ug;
		ug.agentId = agents[index].id;
		ug.cellId = getIdFromCell(getGridCell(NNRADIUS, agents[index]));
		uglist[index] = ug;
	}
}

/*******************
* Helper functions *
********************/

__host__ __device__ float det2(glm::vec3 a, glm::vec3 b){
	// Determinant of 2 vectors (only uses the 2D, x and y components)
	// + result means b is on the left of a, - means otherwise
	return a.x*b.y - a.y*b.x;
}

__host__ __device__ bool intersectRaySphere(glm::vec3 rayStarting, glm::vec3 rayNormalizedDirection, glm::vec3 sphereCenter, float radius2, float& distance){
	float eps = 0.000001f;

	glm::vec3 diff = sphereCenter - rayStarting;
	float t0 = glm::dot(diff, rayNormalizedDirection);
	float dSquared = glm::dot(diff, diff) - t0 * t0;
	if (dSquared > radius2)
	{
		return false;
	}
	float t1 = sqrt(radius2 - dSquared);
	distance = t0 > t1 + eps ? t0 - t1 : t0 + t1;
	return distance > eps;
}

__host__ __device__ glm::vec3 projectPointToLine(glm::vec3 a, glm::vec3 b, glm::vec3 p){
	// A + dot(AP,AB) / dot(AB,AB) * AB
	// http://gamedev.stackexchange.com/questions/72528/how-can-i-project-a-3d-point-onto-a-3d-line

	glm::vec3 ap = p - a;
	glm::vec3 ab = b - a;

	float t = glm::dot(ap, ab) / glm::dot(ab, ab);

	return a + t * ab;
}

__host__ __device__ glm::vec3 projectPointToSegment(glm::vec3 a, glm::vec3 b, glm::vec3 p){
	// A + dot(AP,AB) / dot(AB,AB) * AB
	// http://gamedev.stackexchange.com/questions/72528/how-can-i-project-a-3d-point-onto-a-3d-line

	glm::vec3 ap = p - a;
	glm::vec3 ab = b - a;

	float t = glm::dot(ap, ab) / glm::dot(ab, ab);

	if (t < 0.0){
		return a;
	}
	else if (t > 1.0){
		return b;
	}

	return a + t * ab;
}

__host__ __device__ glm::vec3 projectPointToRay(ray a, glm::vec3 p){
	// Projects a point to its closest location on a ray.
	// If the projected point does not lie on the ray, it snaps to the ray origin
	glm::vec3 ap = p - a.pos;

	float t = glm::dot(ap, glm::normalize(a.dir));

	return a.pos + a.dir * t * float(t >= 0.0);
}

__host__ __device__ glm::vec3 intersectPointRay(ray a, glm::vec3 p){
	// Finds the ray with minimal distance from a point to a ray
	// http://stackoverflow.com/questions/5227373/minimal-perpendicular-vector-between-a-point-and-a-line
	
	return p - (a.pos + (glm::normalize(p-a.pos))*a.dir);
}

__host__ __device__ int sidePointSegment(ray r, glm::vec3 p){
	// Computes what side of a line a point is on
	// http://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
	// If ray pointing up in 2D: +1 is left, 0 is on line, -1 is right

	glm::vec3 a = r.pos;
	glm::vec3 b = r.pos + r.dir;

	return sign((b.x-a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x));
}

__host__ __device__ HRVO computeHRVO(agent A, agent B, float dt){
	HRVO hrvo;
	glm::vec3 pAB = B.pos - A.pos;
	glm::vec3 pABn = glm::normalize(pAB);
	float R = A.radius + B.radius;

	glm::vec3 apex, left, right;

	// Non-colliding
	if (glm::length(pAB) > R){
		apex = (A.vel + B.vel) / 2.0f;

		float theta = glm::asin(R / glm::length(pAB));
		right = glm::rotateZ(pABn, -theta);
		left = glm::rotateZ(pABn, theta);

		// Hybrid RVO, this formulation is taken from Snape HRVO computation
		// Check if vA is on the left of the center line
		float sin2theta = 2.0f * glm::sin(theta) * glm::cos(theta);
		float s;
		if (det2(B.pos - A.pos, A.vel - B.vel) > 0.0f){
			s = 0.5f * det2(A.vel - B.vel, left) / sin2theta;
			apex = B.vel + s * right;
		}
		else {
			s = 0.5f * det2(A.vel - B.vel, right) / sin2theta;
			apex = B.vel + s * left;
		}
	}
	// Colliding
	else {
		apex = 0.5f * (A.vel + B.vel - glm::normalize(pAB) * (R - glm::length(pAB)) / dt);
		right = glm::cross(glm::normalize(pAB), glm::vec3(0.0, 0.0, 1.0));
		left = -right;
	}

	hrvo.apex = apex;
	hrvo.left = left;
	hrvo.right = right;
	return hrvo;

}

/******************
 * initSimulation *
 ******************/

__host__ __device__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

/**
 * Function for generating a random vec3.
 */
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
    thrust::default_random_engine rng(hash((int)(index * time)));
    thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

    return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}


__host__ __device__ glm::vec3 sampleRandom2DVelocity(float time, int index){
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(0, 1);

	float theta = 2.0f*3.1415629f*(float)unitDistrib(rng);
	float r = glm::sqrt((float)unitDistrib(rng));

	return glm::vec3((r*MAX_VEL)*glm::cos(theta), (r*MAX_VEL)*glm::sin(theta), 0.0);
}


/**
 * Initialize memory, update some globals
 */

__global__ void kernInitAgents(int N, agent* agents, float scale, float radius){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		float rad = ((float)index / (float)N) * (2.0f * 3.1415f);

		//if (index == 1){
		//	rad -= 7.0*3.1415f / 8.0f;
		//}

		//agents[index].w = 1.0f / float(index);
		agents[index].w = 1.0f;
		agents[index].pos.x = scale * circle_radius * cos(rad);
		agents[index].pos.y = scale * circle_radius * sin(rad);
		agents[index].pos.z = 0.0;

		/*
		if (index % 2 == 0){
			agents[index].pos.x = circle_radius;
			agents[index].pos.y = circle_radius - 2.0*float(index);
		}
		else{
			agents[index].pos.x = -circle_radius;
			agents[index].pos.y = circle_radius - 2.0*float(index-1);
		}
		*/

		agents[index].goal = -agents[index].pos;

		/*
		agents[index].goal = agents[index].pos;
		agents[index].goal.x *= -1;
		*/

		agents[index].radius = radius;
		agents[index].id = index;
	}
}

void ClearPath::initSimulation(int N) {
	//N = 5;
	numAgents = N;
    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_agents, N*sizeof(agent));
	checkCUDAErrorWithLine("cudaMalloc dev_goals failed!");

	kernInitAgents<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents, scene_scale, robot_radius);
	checkCUDAErrorWithLine("kernInitAgents failed!");

    cudaThreadSynchronize();
}

/******************
 * copyAgentsToVBO *
 ******************/

/**
 * Copy the agent positions into the VBO so that they can be drawn by OpenGL.
 */
__global__ void kernCopyPlanetsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    //float c_scale = -1.0f / s_scale;
	//float c_scale = 1.0f;
	float c_scale = 1.0f / s_scale;

    if (index < N) {
        vbo[4 * index + 0] = pos[index].x * c_scale;
        vbo[4 * index + 1] = pos[index].y * c_scale;
        vbo[4 * index + 2] = pos[index].z * c_scale;
        vbo[4 * index + 3] = 1;
    }
}

/**
 * Wrapper for call to the kernCopyPlanetsToVBO CUDA kernel.
 */
void ClearPath::copyAgentsToVBO(float *vbodptr, glm::vec3* pos, agent* agents, HRVO* hrvos, CandidateVel* candidates, int* neighbors,  int* num_neighbors) {
    dim3 fullBlocksPerGrid((int)ceil(float(numAgents) / float(blockSize)));

    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_pos, vbodptr, scene_scale);
    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");

	cudaMemcpy(pos, dev_pos, numAgents*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(agents, dev_agents, numAgents*sizeof(agent), cudaMemcpyDeviceToHost);
	
	// UG
	cudaMemcpy(neighbors, dev_ug_neighbors, numAgents*numNeighbors*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(num_neighbors, dev_num_neighbors, numAgents*sizeof(int), cudaMemcpyDeviceToHost);

	// HRVOs
	//cudaMemcpy(hrvos, dev_hrvos, totHRVOs*sizeof(HRVO), cudaMemcpyDeviceToHost);
	//cudaMemcpy(candidates, dev_candidates, totCandidates*sizeof(CandidateVel), cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
}

/******************
* Common VO Kernels *
******************/


__global__ void kernUpdatePos(int N, float dt, agent* dev_agents, glm::vec3 *dev_pos){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		// Update positions
		dev_agents[index].pos = dev_agents[index].pos + dev_agents[index].vel * dt;
		dev_pos[index] = dev_agents[index].pos;
	}
}

__global__ void kernUpdateVel(int numAgents, int* agent_ids, agent* agents, bool* in_pcr, glm::vec3* vel_new){
	int index = (blockDim.x*blockIdx.x) + threadIdx.x;

	if (index < numAgents){
		if (in_pcr[index]){
			int agent_id = agent_ids[index];
			agents[agent_id].vel = vel_new[index];
		}
	}
}

__global__ void kernUpdateDesVel(int N, agent *dev_agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		// Get desired velocity
		dev_agents[index].vel = glm::normalize(dev_agents[index].goal - dev_agents[index].pos)*desired_speed;
		float dist = glm::distance(dev_agents[index].goal, dev_agents[index].pos);
		if (dist < 0.1){
			dev_agents[index].vel = glm::vec3(0.0);
		}
	}
}

__global__ void kernComputeNeighbors(int N, int* neighbors, agent* agents){
	// TODO: limit neighbors based on distance from agents
	// TODO: track number of neighbors each agent actually has and populate it that way
	// N - number of agents
	// row-major collapsed matrix - n agents x n-1 neighbors

	// index is the current agent
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < N){
		int numNeighbors = N - 1;
		// i represents id of the neighbor
		for (int i = 0; i < index; i++){
			neighbors[i + index*numNeighbors] = i;
		}
		for (int i = index + 1; i < N; i++){
			neighbors[i - 1 + index*numNeighbors] = i;
		}
	}
}

__global__ void kernComputeNeighbors(int numAgents, int* neighbors, int* num_neighbors, agent* agents){
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < numAgents){

	}
}

__global__ void kernUpdateDesVelCollision(int N, agent* agents, int* neighbors){
	// Deprecated
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		int max_neighbors = N - 1;
		for (int i = 0; i < max_neighbors; i++){
			int n = neighbors[index*max_neighbors + i];
			if (glm::distance(agents[n].pos, agents[index].pos) <= agents[n].radius + agents[index].radius + 1.0){
				agents[index].vel = glm::vec3(0.0);
			}
		}
	}
}

/******************
* HRVO Kernels *
******************/

__global__ void kernComputeHRVOs(int totHRVOs, int numHRVOs, int numAgents, HRVO* hrvos, int* agent_ids, agent* agents, int* neighbors, float dt){
	// Get all RVOs in parallel
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;

		int agent_id = agent_ids[r];

		//hrvos[r*numHRVOs + c] = computeHRVO(agents[agent_id], agents[neighbors[r*numHRVOs + c]], dt);
		hrvos[index] = computeHRVO(agents[agent_id], agents[neighbors[index]], dt);

		printf("%d --> %d\n",agent_id,neighbors[index]);
	}
}

__global__ void kernInitCandidateVels(int totCandidates, CandidateVel* candidates){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totCandidates){
		candidates[index].valid = false;
	}
}

__global__ void kernComputeNaiveCandidateVels(int totHRVOs, int numAgents, int numHRVOs, int numCandidates, CandidateVel* candidates, HRVO* hrvos, int* agent_ids, agent* agents){
	// This computation is heavily borrowed from Snape's HRVO implementation https://github.com/snape/HRVO
	// naive projection velocities
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;
		
		int agent_id = agent_ids[r];

		glm::vec3 prefVel = agents[agent_id].vel;

		float dot1 = glm::dot(prefVel - hrvos[index].apex, hrvos[index].right);
		float dot2 = glm::dot(prefVel - hrvos[index].apex, hrvos[index].left);
		
		CandidateVel candidate;
		candidate.hrvo1 = c;
		candidate.hrvo2 = c;

		if (dot1 > 0.0f && det2(hrvos[index].right, prefVel-hrvos[index].apex) > 0.0f){
			candidate.vel = hrvos[index].apex + dot1*hrvos[index].right;
			candidate.distToPref = glm::length(prefVel - candidate.vel);
			candidate.valid = true;
			candidates[r*numCandidates + 2 * c] = candidate;
		}

		if (dot2 > 0.0f && det2(hrvos[index].left, prefVel - hrvos[index].apex) < 0.0f){
			candidate.vel = hrvos[index].apex + dot2*hrvos[index].left;
			candidate.distToPref = glm::length(prefVel - candidate.vel);
			candidate.valid = true;
			candidates[r*numCandidates + 2 * c + 1] = candidate;
		}

	}
}

__global__ void kernComputeIntersectionCandidateVels(int totHRVOs, int numAgents, int numHRVOs, int numCandidates, int offset, CandidateVel* candidates, HRVO* hrvos, int* agent_ids, agent* agents){
	// This computation is heavily borrowed from Snape's HRVO implementation https://github.com/snape/HRVO
	// offset is usually the number of naiveCandidates (offset of starting location in the candidate buffer
	// intersection projection velocities
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;

		int agent_id = agent_ids[r];

		glm::vec3 prefVel = agents[agent_id].vel;

		CandidateVel candidate;

		int n = 0;
		for (int i = 0; i < numHRVOs; i++){
			if (c == i) continue;
			candidate.hrvo1 = c;
			candidate.hrvo2 = i;

			//hrvos[index] vs hrvos[r*numHRVOs+i]
			int j = r*numHRVOs + i;
			float s, t;

			// Find intersection of right and right
			float d = det2(hrvos[index].right, hrvos[j].right);

			if (d != 0.0f){
				s = det2(hrvos[j].apex - hrvos[index].apex, hrvos[j].right) / d;
				t = det2(hrvos[j].apex - hrvos[index].apex, hrvos[index].right) / d;
				
				if (s >= 0.0f && t >= 0.0f){
					candidate.vel = hrvos[index].apex + s*hrvos[index].right;
					candidate.distToPref = glm::length(prefVel - candidate.vel);
					candidate.valid = true;
					candidates[r*numCandidates + offset + (4*(numHRVOs-1))*c + 4*n] = candidate;
				}
			}

			// Find intersection of left and right
			d = det2(hrvos[index].left, hrvos[j].right);

			if (d != 0.0f){
				s = det2(hrvos[j].apex - hrvos[index].apex, hrvos[j].right) / d;
				t = det2(hrvos[j].apex - hrvos[index].apex, hrvos[index].left) / d;

				if (s >= 0.0f && t >= 0.0f){
					candidate.vel = hrvos[index].apex + s*hrvos[index].left;
					candidate.distToPref = glm::length(prefVel - candidate.vel);
					candidate.valid = true;
					candidates[r*numCandidates + offset + (4 * (numHRVOs - 1))*c + 4 * n + 1] = candidate;
				}
			}

			// Find intersection of right and left
			d = det2(hrvos[index].right, hrvos[j].left);

			if (d != 0.0f){
				s = det2(hrvos[j].apex - hrvos[index].apex, hrvos[j].left) / d;
				t = det2(hrvos[j].apex - hrvos[index].apex, hrvos[index].right) / d;

				if (s >= 0.0f && t >= 0.0f){
					candidate.vel = hrvos[index].apex + s*hrvos[index].right;
					candidate.distToPref = glm::length(prefVel - candidate.vel);
					candidate.valid = true;
					candidates[r*numCandidates + offset + (4 * (numHRVOs - 1))*c + 4 * n + 2] = candidate;
				}
			}

			// Find intersection of left and left
			d = det2(hrvos[index].left, hrvos[j].left);

			if (d != 0.0f){
				s = det2(hrvos[j].apex - hrvos[index].apex, hrvos[j].left) / d;
				t = det2(hrvos[j].apex - hrvos[index].apex, hrvos[index].left) / d;

				if (s >= 0.0f && t >= 0.0f){
					candidate.vel = hrvos[index].apex + s*hrvos[index].left;
					candidate.distToPref = glm::length(prefVel - candidate.vel);
					candidate.valid = true;
					candidates[r*numCandidates + offset + (4 * (numHRVOs - 1))*c + 4 * n + 3] = candidate;
				}
			}

			n++;
		}
	}
}

__global__ void kernComputeInPCR(int numAgents, int numHRVOs, bool* in_pcr, HRVO* hrvos, int* agent_ids, agent* agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numAgents){
		int agent_id = agent_ids[index];

		for (int i = 0; i < numHRVOs; i++){
			if (det2(hrvos[index*numHRVOs + i].left, agents[agent_id].vel - hrvos[index*numHRVOs + i].apex) < 0.0f &&
				det2(hrvos[index*numHRVOs + i].right, agents[agent_id].vel - hrvos[index*numHRVOs + i].apex) > 0.0f){
				in_pcr[index] = true;
			}
		}
	}
}

__global__ void kernComputeValidCandidateVels(int totCandidates, int numAgents, int numHRVOs, int numCandidates, CandidateVel* candidates, HRVO* hrvos, agent* agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totCandidates && candidates[index].valid){
		int r = index / numCandidates;
		int c = index % numCandidates;

		for (int i = 0; i < numHRVOs; i++){
			if (candidates[index].hrvo1 != i &&
				candidates[index].hrvo2 != i &&
				det2(hrvos[r*numHRVOs + i].left, candidates[index].vel - hrvos[r*numHRVOs + i].apex) < 0.0f &&
				det2(hrvos[r*numHRVOs + i].right, candidates[index].vel - hrvos[r*numHRVOs + i].apex) > 0.0f){
				candidates[index].valid = false;
				break;
			}
		}
	}
}

__global__ void kernComputeBestVel(int numAgents, int numCandidates, glm::vec3* vels, CandidateVel* candidates){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < numAgents){

		float minDist = INFINITY;
		glm::vec3 minVel;

		for (int i = 0; i < numCandidates; i++){
			if (candidates[numCandidates*index + i].valid && candidates[numCandidates*index + i].distToPref < minDist){
				minDist = candidates[numCandidates*index + i].distToPref;
				minVel = candidates[numCandidates*index + i].vel;
			}
		}

		vels[index] = minVel;
	}
}

__global__ void kernIndicateNeighborCount(int numAgents, int des_neighbor_count, bool* indicators, int* num_neighbors){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numAgents){
		indicators[index] = false;

		if (num_neighbors[index] == des_neighbor_count){
			indicators[index] = true;
		}
	}
}

__global__ void kernGetSubsetAgentIds(int num_sub_agents, int* sub_agent_ids, int* idx, agent* agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_sub_agents){
		sub_agent_ids[index] = agents[idx[index]].id;
	}
}

__global__ void kernGetSubsetNeighborIds(int num_sub_neighbors, int num_sub_neighbors_per_agent, int num_max_neighbors_per_agent, int* sub_neighbors, int* neighbors, int* idx){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_sub_neighbors){
		int r = index / num_sub_neighbors_per_agent;
		int c = index % num_sub_neighbors_per_agent;

		int agent_id = idx[r];
		sub_neighbors[index] = neighbors[agent_id*num_max_neighbors_per_agent + c];
	}
}

__global__ void kernGetSubsetIdxes(int numAgents, int* idxes, bool* indicators, int* indicator_sum, agent* agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numAgents){
		if (indicators[index]){
			idxes[indicator_sum[index]] = agents[index].id;
		}
	}
}


/******************
* Uniform Grid Kernels *
******************/

__global__ void kernInitStartIdxes(int numIdx, int* startIdx){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numIdx){
		startIdx[index] = -1;
	}
}

__global__ void kernComputeUGNeighbors(int numAgents, int max_num_neighbors, int* neighbors, int* max_neighbors, agent* agents, int* startIdx, UGEntry* ug_list){
	// numAgents - number of agents
	// max_num_neighbors - usually numAgents - 1, but the max number of neighbors we are considering
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numAgents){
		Cell cell = getGridCell(NNRADIUS, agents[index]);
		int dxs[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
		int dys[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

		int max_n = 0;

		Cell ncell;
		for (int i = 0; i < 9; i++){
			int dx = dxs[i];
			int dy = dys[i];

			ncell.x = cell.x + dx;
			ncell.y = cell.y + dy;

			if (ncell.x < 0 || ncell.y < 0 || ncell.x >= GRIDMAX || ncell.y >= GRIDMAX){
				continue;
			}

			int ncellId = getIdFromCell(ncell);

			int start = startIdx[ncellId];
			if (start < 0) continue; // No agents in this cell

			for (int j = start; j < numAgents; j++){
				int oagentId = ug_list[j].agentId;
				if (oagentId == index) continue; // Skip if it's myself
				if (ug_list[j].cellId != ncellId) break; // Hit the end of these

				if (glm::distance(agents[oagentId].pos, agents[index].pos) < NNRADIUS){
					neighbors[index*max_num_neighbors + max_n] = oagentId;
					max_n++;
				}
			}
		}
		max_neighbors[index] = max_n;
	}
}

__global__ void kernUGStartIdxes(int numAgents, int* startIdx, UGEntry* ug_list){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numAgents){
		int idx = ug_list[index].cellId;

		if (index == 0){
			startIdx[idx] = index;
		}
		else {
			int idx_prev = ug_list[index - 1].cellId;
			// Note: Race condition happens if we do not block off other threads from writing to here
			if (idx_prev != idx){
				startIdx[idx] = index;
			}
		}
	}
}

/******************
 * stepSimulation *
 ******************/

/**
 * Step the entire N-body simulation by `dt` seconds.
 */
void ClearPath::stepSimulation(float dt, int iter) {

	dim3 fullBlocksPerGrid((numAgents + blockSize - 1) / blockSize);

	// Update all the desired velocities given current positions
	kernUpdateDesVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents);

	// Allocate space for neighbors, FVOs, intersection points (how to do this?)
	numNeighbors = numAgents - 1;

	// Get the number of blocks we need
	dim3 fullBlocksForUG((GRIDMAX*GRIDMAX + blockSize - 1) / blockSize);

	// Free everything
	cudaFree(dev_neighbors);
	
	// UG
	cudaFree(dev_uglist);
	cudaFree(dev_startIdx);
	cudaFree(dev_ug_neighbors);
	cudaFree(dev_num_neighbors);

	cudaFree(dev_indicators);
	cudaFree(dev_indicator_sum);

	cudaMalloc((void**)&dev_neighbors, numAgents*numNeighbors*sizeof(int));

	// UG
	cudaMalloc((void**)&dev_uglist, numAgents*sizeof(UGEntry));
	cudaMalloc((void**)&dev_startIdx, GRIDMAX*GRIDMAX*sizeof(int));
	cudaMalloc((void**)&dev_ug_neighbors, numNeighbors*numAgents*sizeof(int));
	cudaMalloc((void**)&dev_num_neighbors, numAgents*sizeof(int));

	cudaMalloc((void**)&dev_indicators, numAgents*sizeof(bool));
	cudaMalloc((void**)&dev_indicator_sum, numAgents*sizeof(int));

	// Find neighbors
	// TODO: 2 arrays: 1 of all the ones that have same # neighbors, other has the remaining uncomputed ones
	kernComputeNeighbors<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_neighbors, dev_agents);
	cudaThreadSynchronize();

	// Compute Neighbors with the Uniform Grid (UG)
	kernUpdateUniformGrid<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_uglist, dev_agents);
	//cudaThreadSynchronize();

	// Sort by cell id
	thrust::device_ptr<UGEntry> thrust_uglist = thrust::device_pointer_cast(dev_uglist);
	thrust::sort(thrust::device,thrust_uglist, thrust_uglist+numAgents, UGComp());

	// Create start index structure
	kernInitStartIdxes<<<fullBlocksForUG, blockSize>>>(GRIDMAX*GRIDMAX, dev_startIdx);
	kernUGStartIdxes<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_startIdx, dev_uglist);

	// Get the neighbors
	kernComputeUGNeighbors<<<fullBlocksPerGrid, blockSize>>>(numAgents, numNeighbors, dev_ug_neighbors, dev_num_neighbors, dev_agents, dev_startIdx, dev_uglist);

	int num_sub_agents;
	bool last_indicator;

	// Iterate through the number of neighbors and update agents accordingly
	for (int n = 1; n <= numNeighbors; n++){

		// Get current neighbors
		kernIndicateNeighborCount<<<fullBlocksPerGrid, blockSize>>>(numAgents, n, dev_indicators, dev_num_neighbors);
		thrust::exclusive_scan(thrust::device, dev_indicators, dev_indicators+numAgents, dev_indicator_sum);

		checkCUDAErrorWithLine("thrust failed?");

		num_sub_agents = 0;
		last_indicator = false;
		cudaMemcpy(&last_indicator, dev_indicators + numAgents-1, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&num_sub_agents, dev_indicator_sum + numAgents-1, sizeof(int), cudaMemcpyDeviceToHost);
		num_sub_agents += last_indicator;

		checkCUDAErrorWithLine("computing numbers failed?");

		if (num_sub_agents == 0) continue;

		printf("---num_neighbors: %d, num_agent: %d---\n", n, num_sub_agents);

		cudaFree(dev_hrvos);
		cudaFree(dev_neighbor_ids);
		cudaFree(dev_agent_ids);
		cudaFree(dev_candidates);
		cudaFree(dev_in_pcr);
		cudaFree(dev_vel_new);
		cudaFree(dev_idx);

		cudaMalloc((void**)&dev_idx, num_sub_agents*sizeof(int));

		kernGetSubsetIdxes<<<fullBlocksPerGrid,blockSize>>>(numAgents, dev_idx, dev_indicators, dev_indicator_sum, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc subset idxes failed!");

		cudaMalloc((void**)&dev_agent_ids, num_sub_agents*sizeof(int));
		cudaMalloc((void**)&dev_neighbor_ids, num_sub_agents*n*sizeof(int));

		dim3 fullBlocksForNeighborIds((n*num_sub_agents + blockSize - 1) / blockSize);
		dim3 fullBlocksForAgentIds((num_sub_agents + blockSize - 1) / blockSize);

		kernGetSubsetAgentIds<<<fullBlocksForAgentIds, blockSize>>>(num_sub_agents, dev_agent_ids, dev_idx, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc subsetagentsid failed!");

		kernGetSubsetNeighborIds<<<fullBlocksForNeighborIds, blockSize>>>(num_sub_agents*n, n, numNeighbors, dev_neighbor_ids, dev_ug_neighbors, dev_idx);
		checkCUDAErrorWithLine("cudaMalloc subsetneighborsid failed!");

		int* hst_neighbor_ids = (int*)malloc(sizeof(int)*num_sub_agents*n);
		cudaMemcpy(hst_neighbor_ids, dev_neighbor_ids, sizeof(int)*num_sub_agents*n, cudaMemcpyDeviceToHost);
		printf("things: ");
		for (int jj = 0; jj < n*num_sub_agents; jj++){
			printf("%d ",hst_neighbor_ids[jj]);
		}
		printf("\n");
		free(hst_neighbor_ids);

		numHRVOs = n;
		totHRVOs = num_sub_agents*n;
		int numNaiveCandidates = 2 * numHRVOs;
		int numIntersectionCandidates = 4 * numHRVOs * (numHRVOs - 1);
		numCandidates = numNaiveCandidates + numIntersectionCandidates;
		totCandidates = num_sub_agents * numCandidates;

		dim3 fullBlocksForHRVOs((totHRVOs + blockSize - 1) / blockSize);
		dim3 fullBlocksForCandidates((totCandidates + blockSize - 1) / blockSize);

		cudaMalloc((void**)&dev_hrvos, totHRVOs*sizeof(HRVO));
		cudaMalloc((void**)&dev_candidates, totCandidates*sizeof(CandidateVel));
		cudaMalloc((void**)&dev_vel_new, num_sub_agents*sizeof(glm::vec3));
		cudaMalloc((void**)&dev_in_pcr, num_sub_agents*sizeof(bool));

		cudaMemset(dev_in_pcr, false, num_sub_agents*sizeof(bool));
		cudaThreadSynchronize();

		kernComputeHRVOs<<<fullBlocksPerGrid, blockSize>>>(totHRVOs, numHRVOs, num_sub_agents, dev_hrvos, dev_agent_ids, dev_agents, dev_neighbor_ids, dt);
		checkCUDAErrorWithLine("cudaMalloc dev_goals failed!");

		kernInitCandidateVels<<<fullBlocksForCandidates, blockSize>>>(totCandidates, dev_candidates);
		checkCUDAErrorWithLine("cudaMalloc cand vels failed!");

		kernComputeNaiveCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, num_sub_agents, numHRVOs, numCandidates, dev_candidates, dev_hrvos, dev_agent_ids, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc naive vels failed!");

		kernComputeIntersectionCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, num_sub_agents, numHRVOs, numCandidates, numNaiveCandidates, dev_candidates, dev_hrvos, dev_agent_ids, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc intersection vels failed!");

		kernComputeValidCandidateVels<<<fullBlocksForCandidates, blockSize>>>(totCandidates, num_sub_agents, numHRVOs, numCandidates, dev_candidates, dev_hrvos, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc compute valid vels failed!");

		kernComputeBestVel<<<fullBlocksPerGrid, blockSize>>>(num_sub_agents, numCandidates, dev_vel_new, dev_candidates);
		checkCUDAErrorWithLine("cudaMalloc best vels failed!");

		kernComputeInPCR<<<fullBlocksPerGrid, blockSize>>>(num_sub_agents, numHRVOs, dev_in_pcr, dev_hrvos, dev_agent_ids, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc in pcr failed!");

		kernUpdateVel<<<fullBlocksPerGrid, blockSize>>>(num_sub_agents, dev_agent_ids, dev_agents, dev_in_pcr, dev_vel_new);
		checkCUDAErrorWithLine("cudaMalloc update vels failed!");

	}

	agent* hst_agents = (agent*)malloc(numAgents*sizeof(agent));
	cudaMemcpy(hst_agents, dev_agents, numAgents*sizeof(agent), cudaMemcpyDeviceToHost);
	free(hst_agents);

	// Update the positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_pos);


}
