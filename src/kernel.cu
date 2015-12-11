#define GLM_FORCE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
//#include <thrust/unique.h>
//#include <thrust/copy.h>
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

//#include <chrono>
//#include <vector>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define sign(x) (x>0)-(x<0)

#define CUDA_UG
//#define CUDA_NN
//#define CPU_UG
//#define CPU_NN

//#define INFINITY 0x7f800000
#define NEG_INFINITY 0xff800000

//#define MAX_NEIGHBORS 10

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
#define blockSize 1024 //128
#define robot_radius 0.5 // 0.5 default
#define circle_radius 40 // 10 for 30 robots, 30 for 100 robots
#define desired_speed 3.0f // 3.0 default

#define GRIDMAX 50 // Must be even, creates grid of GRIDMAX x GRIDMAX size

#define NNRADIUS 3.0f // 2.0 default, 1.5 for 100 robots

// Experimental sampling
#define MAX_VEL 4.0f // 3.0 default

/**********************************************
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
//int* dev_neighbors; //index of neighbors
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

bool* dev_unique_indicators;
int* dev_unique_idxes;
int* dev_unique_counts;

HRVO* dev_hrvos;
CandidateVel* dev_candidates;


agent* hst_agents_;
UGEntry* hst_uglist_;
int* hst_startIdx_;
int* hst_ug_neighbors_;
int* hst_num_neighbors_;

/*************************
* Uniform Grid Functions *
**************************/

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



/*******************
* Helper functions *
********************/

__host__ __device__ float sqr(glm::vec3 a){
	return glm::dot(a, a);
}

__host__ __device__ float det2(glm::vec3 a, glm::vec3 b){
	// Determinant of 2 vectors (only uses the 2D, x and y components)
	// + result means b is on the left of a, - means otherwise
	return a.x*b.y - a.y*b.x;
}

__host__ __device__ bool intersectRaySphere(glm::vec3 rayStarting, glm::vec3 rayNormalizedDirection, glm::vec3 sphereCenter, float radius2, float& distance1, float& distance2){
	// Modified from the glm version to work on device
	float eps = 0.000001f;

	glm::vec3 diff = sphereCenter - rayStarting;
	float t0 = glm::dot(diff, rayNormalizedDirection);
	float dSquared = glm::dot(diff, diff) - t0 * t0;
	if (dSquared > radius2)
	{
		return false;
	}
	float t1 = sqrt(radius2 - dSquared);

	distance1 = t0 - t1;
	distance2 = t0 + t1;

	float distance = t0 > t1 + eps ? t0 - t1 : t0 + t1;
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

__host__ __device__ glm::vec3 projectPointToRay(Ray a, glm::vec3 p){
	// Projects a point to its closest location on a ray.
	// If the projected point does not lie on the ray, it snaps to the ray origin
	glm::vec3 ap = p - a.pos;

	float t = glm::dot(ap, glm::normalize(a.dir));

	return a.pos + a.dir * t * float(t >= 0.0);
}

__host__ __device__ glm::vec3 intersectPointRay(Ray a, glm::vec3 p){
	// Finds the ray with minimal distance from a point to a ray
	// http://stackoverflow.com/questions/5227373/minimal-perpendicular-vector-between-a-point-and-a-line
	
	return p - (a.pos + (glm::normalize(p-a.pos))*a.dir);
}

__host__ __device__ bool intersectRayRay(Ray a, Ray b, glm::vec3& point){
	// Solves p1 + t1*v1 = p2 + t2*v2
	// [t1; t2] = [v1x -v2x; v1y -v2y]^-1*(p2-p1);
	bool isIntersection = false;

	// Parallel lines cannot intersect
	if (a.dir.x == b.dir.x && a.dir.y == b.dir.y){
		return false;
	}

	glm::vec2 ts;

	ts = glm::inverse(glm::mat2(a.dir.x, a.dir.y, -b.dir.x, -b.dir.y)) * glm::vec2(b.pos - a.pos);

	if (ts.x >= 0 && ts.y >= 0){
		point = glm::vec3(a.pos + a.dir*ts.x);
		point.z = 0.0;
		return true;
		
	}
	return false;
	
}

__host__ __device__ int sidePointSegment(Ray r, glm::vec3 p){
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

__host__ __device__ float sampleRandomSpeed(float time, int index){
	thrust::default_random_engine rng(hash((int)((index+1) * time)));
	thrust::uniform_real_distribution<float> unitDistrib(0, 1);

	return MAX_VEL * (float)unitDistrib(rng);
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
		//float rad = ((float)(index%20) / (float)20) * (2.0f * 3.1415f);
		
		//if (index == 1){
		//	rad -= 7.0*3.1415f / 8.0f;
		//}

		//agents[index].w = 1.0f / float(index);
		agents[index].w = 1.0f;
		agents[index].pos.x = scale * circle_radius * cos(rad);
		agents[index].pos.y = scale * circle_radius * sin(rad);
		agents[index].pos.z = 0.0;

		// lines
		/*
		if (index % 2 == 0){
			agents[index].pos.x = circle_radius;
			agents[index].pos.y = circle_radius - 1.1*float(index);
		}
		else{
			agents[index].pos.x = -circle_radius;
			agents[index].pos.y = circle_radius - 1.1*float(index-1);
		}
		*/

		agents[index].goal = -agents[index].pos;

		// lines
		/*
		agents[index].goal = agents[index].pos;
		agents[index].goal.x *= -1;
		*/

		agents[index].radius = radius;
		agents[index].id = index;
		
		/*
		agents[index].pos.x -= 30; //30 btwn circles
		agents[index].pos.y -= 30;
		agents[index].goal.x -= 30; //30 btwn circles
		agents[index].goal.y -= 30;

		int shift = index / 20;
		int mul = 20;
		agents[index].pos.x += shift * mul;
		agents[index].pos.y += shift * mul;
		agents[index].goal.x += shift * mul;
		agents[index].goal.y += shift * mul;
		*/
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

/*******************
 * copyAgentsToVBO *
 *******************/

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

/*************************
* Common Utility Kernels *
**************************/

struct is_true{
	__host__ __device__ bool operator()(const bool x){
		return x;
	}
};

__global__ void kernUnique(int N, bool* indicators, int* input){
	// Expects a sorted array of integers, returns a bool array of the same length indicating 
	// whether or not a position in that array contains a unique integer

	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < N){
		if (index == 0 || input[index] != input[index-1]){
			indicators[index] = true;
			return;
		}
		indicators[index] = false;
	}
}

__global__ void kernGather(int N, int* output, int* input, bool* indicators, int* idxes){
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < N){
		if (indicators[index]){
			output[idxes[index]] = input[index];
		}
	}
}

/********************
* Common VO Kernels *
*********************/


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

__global__ void kernUpdateDesVel(int N, agent *dev_agents, int iter){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		// Get desired velocity
		float randomSpeed = sampleRandomSpeed((float)iter, index);
		dev_agents[index].vel = glm::normalize(dev_agents[index].goal - dev_agents[index].pos)*randomSpeed;
		float dist = glm::distance(dev_agents[index].goal, dev_agents[index].pos);
		if (dist < 0.1){
			dev_agents[index].vel = glm::vec3(0.0);
		}
	}
}

__global__ void kernComputeNeighbors(int N, int* neighbors, int* num_neighbors, agent* agents){
	// TODO: limit neighbors based on distance from agents
	// TODO: track number of neighbors each agent actually has and populate it that way
	// N - number of agents
	// row-major collapsed matrix - n agents x n-1 neighbors

	// index is the current agent
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < N){
		/*
		int numNeighbors = N - 1;
		// i represents id of the neighbor
		for (int i = 0; i < index; i++){
			neighbors[i + index*numNeighbors] = i;
		}
		for (int i = index + 1; i < N; i++){
			neighbors[i - 1 + index*numNeighbors] = i;
		}
		num_neighbors[index] = numNeighbors;
		*/

		int maxNeighbors = N - 1;
		int numNeighbors = 0;
		// i represents id of the neighbor
		for (int i = 0; i < index; i++){
			if (glm::length(agents[index].pos - agents[i].pos) < NNRADIUS){
				neighbors[numNeighbors + index*maxNeighbors] = i;
				numNeighbors++;
			}
		}
		for (int i = index + 1; i < N; i++){
			if (glm::length(agents[index].pos - agents[i].pos) < NNRADIUS){
				neighbors[numNeighbors + index*maxNeighbors] = i;
				numNeighbors++;
			}
		}
		num_neighbors[index] = numNeighbors;
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

/***************
* HRVO Kernels *
****************/

__global__ void kernComputeHRVOs(int totHRVOs, int numHRVOs, int numAgents, HRVO* hrvos, int* agent_ids, agent* agents, int* neighbors, float dt){
	// Get all RVOs in parallel
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;

		int agent_id = agent_ids[r];

		//hrvos[r*numHRVOs + c] = computeHRVO(agents[agent_id], agents[neighbors[r*numHRVOs + c]], dt);
		hrvos[index] = computeHRVO(agents[agent_id], agents[neighbors[index]], dt);

		//printf("%d --> %d\n",agent_id,neighbors[index]);
	}
}

__global__ void kernInitCandidateVels(int totCandidates, CandidateVel* candidates){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totCandidates){
		candidates[index].valid = false;
	}
}

__global__ void kernComputeNaiveCandidateVels(int totHRVOs, int numAgents, int numHRVOs, int numCandidates, CandidateVel* candidates, HRVO* hrvos, int* agent_ids, agent* agents){
	// Project the preferred velocity onto the HRVO to get candidate velocities
	// naive projection velocities
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;
		
		int agent_id = agent_ids[r];

		glm::vec3 prefVel = agents[agent_id].vel;

		CandidateVel candidate;
		candidate.hrvo1 = c;
		candidate.hrvo2 = c;

		Ray right;
		right.pos = hrvos[index].apex;
		right.dir = hrvos[index].right;
		candidate.vel = projectPointToRay(right, prefVel);
		candidate.distToPref = sqr(prefVel - candidate.vel);
		candidate.valid = true;
		if (glm::length(candidate.vel) < MAX_VEL) candidates[r*numCandidates + 2 * c] = candidate;

		Ray left;
		left.pos = hrvos[index].apex;
		left.dir = hrvos[index].left;
		candidate.vel = projectPointToRay(right, prefVel);
		candidate.distToPref = sqr(prefVel - candidate.vel);
		candidate.valid = true;
		if (glm::length(candidate.vel) < MAX_VEL) candidates[r*numCandidates + 2 * c] = candidate;

	}
}

__global__ void kernComputeMaxCandidateVels(int totHRVOs, int numAgents, int numHRVOs, int numCandidates, int offset, CandidateVel* candidates, HRVO* hrvos, int* agent_ids, agent* agents){
	// Intersect circle around each agent with HRVOS (get max velocities along the HRVOs)

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;

		int agent_id = agent_ids[r];

		glm::vec3 prefVel = agents[agent_id].vel;

		CandidateVel candidate;
		candidate.hrvo1 = INT_MAX;
		candidate.hrvo2 = c;

		float discriminant = MAX_VEL * MAX_VEL - det2(hrvos[index].apex, hrvos[index].right)*det2(hrvos[index].apex, hrvos[index].right);

		if (discriminant > 0.0f) {

			float t1 = -(glm::dot(hrvos[index].apex, hrvos[index].right)) + glm::sqrt(discriminant);
			float t2 = -(glm::dot(hrvos[index].apex, hrvos[index].right)) - std::sqrt(discriminant);

			if (t1 >= 0.0f) {
				candidate.vel = hrvos[index].apex + t1*hrvos[index].right;
				candidate.distToPref = sqr(candidate.vel - prefVel);
				candidate.valid = true;
				candidates[r*numCandidates + offset + 4*c] = candidate;
			}

			if (t2 >= 0.0f) {
				candidate.vel = hrvos[index].apex + t2*hrvos[index].right;
				candidate.distToPref = sqr(candidate.vel - prefVel);
				candidate.valid = true;
				candidates[r*numCandidates + offset + 4 * c + 1] = candidate;
			}
		}

		discriminant = MAX_VEL * MAX_VEL - det2(hrvos[index].apex, hrvos[index].left)*det2(hrvos[index].apex, hrvos[index].left);

		if (discriminant > 0.0f) {

			float t1 = -(glm::dot(hrvos[index].apex, hrvos[index].left)) + glm::sqrt(discriminant);
			float t2 = -(glm::dot(hrvos[index].apex, hrvos[index].left)) - std::sqrt(discriminant);

			if (t1 >= 0.0f) {
				candidate.vel = hrvos[index].apex + t1*hrvos[index].left;
				candidate.distToPref = sqr(candidate.vel - prefVel);
				candidate.valid = true;
				candidates[r*numCandidates + offset + 4 * c + 2] = candidate;
			}

			if (t2 >= 0.0f) {
				candidate.vel = hrvos[index].apex + t2*hrvos[index].left;
				candidate.distToPref = sqr(candidate.vel - prefVel);
				candidate.valid = true;
				candidates[r*numCandidates + offset + 4 * c + 3] = candidate;
			}
		}
	}
}

__global__ void kernComputeIntersectionCandidateVels(int totHRVOs, int numAgents, int numHRVOs, int numCandidates, int offset, CandidateVel* candidates, HRVO* hrvos, int* agent_ids, agent* agents){
	// Computes the intersections between all the HRVOs
	// offset is usually the number of naiveCandidates (offset of starting location in the candidate buffer
	// intersection projection velocities
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;

		int agent_id = agent_ids[r];

		glm::vec3 prefVel = agents[agent_id].vel;

		CandidateVel candidate;

		Ray curr_left;
		curr_left.pos = hrvos[index].apex;
		curr_left.dir = hrvos[index].left;

		Ray curr_right;
		curr_right.pos = hrvos[index].apex;
		curr_right.dir = hrvos[index].right;

		bool intersects;
		int n = 0;
		for (int i = 0; i < numHRVOs; i++){
			if (c == i) continue;
			candidate.hrvo1 = c;
			candidate.hrvo2 = i;

			int j = r*numHRVOs + i;
			float s, t;

			Ray left;
			left.pos = hrvos[j].apex;
			left.dir = hrvos[j].left;

			Ray right;
			right.pos = hrvos[j].apex;
			right.dir = hrvos[j].right;

			intersects = intersectRayRay(curr_left, left, candidate.vel);
			if (intersects){
				candidate.distToPref = sqr(prefVel - candidate.vel);
				candidate.valid = true;
				if (glm::length(candidate.vel) < MAX_VEL) candidates[r*numCandidates + offset + (4 * (numHRVOs - 1))*c + 4 * n] = candidate;
			}

			intersects = intersectRayRay(curr_left, right, candidate.vel);
			if (intersects){
				candidate.distToPref = sqr(prefVel - candidate.vel);
				candidate.valid = true;
				if (glm::length(candidate.vel) < MAX_VEL) candidates[r*numCandidates + offset + (4 * (numHRVOs - 1))*c + 4 * n + 1] = candidate;
			}

			intersects = intersectRayRay(curr_right, left, candidate.vel);
			if (intersects){
				candidate.distToPref = sqr(prefVel - candidate.vel);
				candidate.valid = true;
				if (glm::length(candidate.vel) < MAX_VEL) candidates[r*numCandidates + offset + (4 * (numHRVOs - 1))*c + 4 * n + 2] = candidate;
			}

			intersects = intersectRayRay(curr_right, right, candidate.vel);
			if (intersects){
				candidate.distToPref = sqr(prefVel - candidate.vel);
				candidate.valid = true;
				if (glm::length(candidate.vel) < MAX_VEL) candidates[r*numCandidates + offset + (4 * (numHRVOs - 1))*c + 4 * n + 3] = candidate;
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


/*********************************
* Nearest Neighbors CPU Versions *
**********************************/

void computeNeighbors(int N, int* neighbors, int* num_neighbors, agent* agents){
	for (int index = 0; index < N; index++){
		int maxNeighbors = N - 1;
		int numNeighbors = 0;
		// i represents id of the neighbor
		for (int i = 0; i < N; i++){
			if (i == index) continue;
			if (glm::length(agents[index].pos - agents[i].pos) < NNRADIUS){
				neighbors[numNeighbors + index*maxNeighbors] = i;
				numNeighbors++;
			}
		}
		num_neighbors[index] = numNeighbors;
	}
}

void updateUniformGrid(int numAgents, UGEntry* uglist, agent* agents){
	for (int index = 0; index < numAgents; index++){
		UGEntry ug;
		ug.agentId = agents[index].id;
		ug.cellId = getIdFromCell(getGridCell(NNRADIUS, agents[index]));
		uglist[index] = ug;
	}
}

void initStartIdxes(int numIdx, int* startIdx){
	for (int i = 0; i < numIdx; i++){
		startIdx[i] = -1;
	}
}

void computeUGNeighbors(int numAgents, int max_num_neighbors, int* neighbors, int* max_neighbors, agent* agents, int* startIdx, UGEntry* ug_list){
	for (int index = 0; index < numAgents; index++){
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

void UGStartIdxes(int numAgents, int* startIdx, UGEntry* ug_list){
	for (int index = 0; index < numAgents; index++){
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

/***********************
* Uniform Grid Kernels *
************************/
__global__ void kernUpdateUniformGrid(int numAgents, UGEntry* uglist, agent* agents){
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < numAgents){
		UGEntry ug;
		ug.agentId = agents[index].id;
		ug.cellId = getIdFromCell(getGridCell(NNRADIUS, agents[index]));
		uglist[index] = ug;
	}
}

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

	glm::vec3 prefVel = glm::vec3(2.5f, 3.0f, 0.0f);
	glm::vec3 apex = glm::vec3(2.0, 2.0, 0.0);
	glm::vec3 left = glm::normalize(glm::vec3(-0.5, 1.0, 0.0));
	glm::vec3 right = glm::normalize(glm::vec3(0.5, 1.0, 0.0));

	float s;
	float t;
	bool intersects = intersectRaySphere(apex, left, glm::vec3(0.0, 0.0, 0.0), 3.0f*3.0f, s, t);



	/*
	glm::vec3 prefVel = glm::vec3(2.5f, 3.0f, 0.0f);
	glm::vec3 apex = glm::vec3(3.0, 2.0, 0.0);
	glm::vec3 left = glm::normalize(glm::vec3(-0.5,1.0,0.0));
	glm::vec3 right = glm::normalize(glm::vec3(0.5, 1.0, 0.0));

	float dot1 = glm::dot(prefVel - apex, right);
	float dot2 = glm::dot(prefVel - apex, left);

	glm::vec3 resultOnRight;
	glm::vec3 resultOnLeft;

	if (dot1 > 0.0f && det2(right, prefVel - apex) > 0.0f){
		resultOnRight = apex + dot1*right;

	}

	if (dot2 > 0.0f && det2(left, prefVel - apex) < 0.0f){
		resultOnLeft = apex + dot2*left;
	}

	printf("right: (%f, %f, %f)\n", resultOnRight.x, resultOnRight.y, resultOnRight.z);
	printf("left: (%f, %f, %f)\n", resultOnLeft.x, resultOnLeft.y, resultOnLeft.z);

	Ray r;
	r.pos = apex;
	r.dir = left;
	glm::vec3 pointOnLeft = projectPointToRay(r, prefVel);
	r.pos = apex;
	r.dir = right;
	glm::vec3 pointOnRight = projectPointToRay(r, prefVel);

	printf("right mine: (%f, %f, %f)\n", pointOnRight.x, pointOnRight.y, pointOnRight.z);
	printf("left mine: (%f, %f, %f)\n", pointOnLeft.x, pointOnLeft.y, pointOnLeft.z);
	*/

	dim3 fullBlocksPerGrid((numAgents + blockSize - 1) / blockSize);

	// Update all the desired velocities given current positions
	//kernUpdateDesVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents);
	kernUpdateDesVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents, iter);

	// Allocate space for neighbors, FVOs, intersection points (how to do this?)
	numNeighbors = numAgents - 1;

	// Get the number of blocks we need
	dim3 fullBlocksForUG((GRIDMAX*GRIDMAX + blockSize - 1) / blockSize);

	// Free everything
	cudaFree(dev_unique_indicators);
	cudaFree(dev_unique_idxes);

	cudaFree(dev_uglist);
	cudaFree(dev_startIdx);
	cudaFree(dev_ug_neighbors);
	cudaFree(dev_num_neighbors);

	cudaFree(dev_indicators);
	cudaFree(dev_indicator_sum);

	// Allocation for Uniform Grid
	cudaMalloc((void**)&dev_uglist, numAgents*sizeof(UGEntry));
	cudaMalloc((void**)&dev_startIdx, GRIDMAX*GRIDMAX*sizeof(int));
	cudaMalloc((void**)&dev_ug_neighbors, numNeighbors*numAgents*sizeof(int));
	cudaMalloc((void**)&dev_num_neighbors, numAgents*sizeof(int));

	cudaMalloc((void**)&dev_indicators, numAgents*sizeof(bool));
	cudaMalloc((void**)&dev_indicator_sum, numAgents*sizeof(int));

	int* dev_neighbor_counts;
	int* dev_neighbor_counts_end;
	cudaMalloc((void**)&dev_neighbor_counts, sizeof(int)*numAgents);



	// Compute Neighbors with GPU Uniform Grid
#ifdef CUDA_UG
	/*
	cudaEvent_t startUG, stopUG;
	cudaEventCreate(&startUG);
	cudaEventCreate(&stopUG);
	cudaEventRecord(startUG);
	*/

	kernUpdateUniformGrid<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_uglist, dev_agents);
	
	// Sort by cell id
	thrust::device_ptr<UGEntry> thrust_uglist = thrust::device_pointer_cast(dev_uglist);
	thrust::sort(thrust::device,thrust_uglist, thrust_uglist+numAgents, UGComp());

	// Create start index structure
	kernInitStartIdxes<<<fullBlocksForUG, blockSize>>>(GRIDMAX*GRIDMAX, dev_startIdx);
	kernUGStartIdxes<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_startIdx, dev_uglist);

	// Get the neighbors
	kernComputeUGNeighbors<<<fullBlocksPerGrid, blockSize>>>(numAgents, numNeighbors, dev_ug_neighbors, dev_num_neighbors, dev_agents, dev_startIdx, dev_uglist);
	
	/*
	cudaEventRecord(stopUG);
	cudaEventSynchronize(stopUG);
	float millisecondsUG = 0;
	cudaEventElapsedTime(&millisecondsUG, startUG, stopUG);
	printf("UG: %f\n", millisecondsUG);
	*/
#endif



#ifdef CUDA_NN
	cudaEvent_t startN, stopN;
	cudaEventCreate(&startN);
	cudaEventCreate(&stopN);
	cudaEventRecord(startN);

	kernComputeNeighbors<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_ug_neighbors, dev_num_neighbors, dev_agents);

	cudaEventRecord(stopN);
	cudaEventSynchronize(stopN);
	float millisecondsN = 0;
	cudaEventElapsedTime(&millisecondsN, startN, stopN);
	printf("No grid: %f\n", millisecondsN);
#endif

#ifdef defined(CPU_UG) || defined(CPU_NN)
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	hst_agents_ = (agent*)malloc(numAgents*sizeof(agent));

	hst_uglist_ = (UGEntry*)malloc(numAgents*sizeof(UGEntry));
	hst_startIdx_ = (int*)malloc(GRIDMAX*GRIDMAX*sizeof(int));
	hst_ug_neighbors_ = (int*)malloc(numNeighbors*numAgents*sizeof(int));
	hst_num_neighbors_ = (int*)malloc(numAgents*sizeof(int));

	cudaMemcpy(hst_agents_, dev_agents, numAgents*sizeof(agent), cudaMemcpyDeviceToHost);
#endif

	// Compute Neighbors with CPU Uniform Grid
#ifdef CPU_UG
	updateUniformGrid(numAgents, hst_uglist_, hst_agents_);
	thrust::sort(thrust::host, hst_uglist_, hst_uglist_+numAgents, UGComp());

	initStartIdxes(GRIDMAX*GRIDMAX, hst_startIdx_);
	UGStartIdxes(numAgents, hst_startIdx_, hst_uglist_);
	/*
	for (int i = 0; i < numAgents; i++){
		printf("(cell %d, robot %d)\n", hst_uglist_[i].cellId, hst_uglist_[i].agentId);
	}

	for (int i = 0; i < GRIDMAX*GRIDMAX; i++){
		if (hst_startIdx_[i] > -1){
			printf("(cell %d, robot %d)\n", i, hst_startIdx_[i]);
		}
	}
	*/
	computeUGNeighbors(numAgents, numNeighbors, hst_ug_neighbors_, hst_num_neighbors_, hst_agents_, hst_startIdx_, hst_uglist_);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("CPU UG: %f\n",elapsed_seconds.count());
#endif

#ifdef CPU_NN
	computeNeighbors(numAgents, hst_ug_neighbors_, hst_num_neighbors_, hst_agents_);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("CPU NN: %f\n", elapsed_seconds.count());
#endif

#ifdef defined(CPU_UG) || defined(CPU_NN)
	cudaMemcpy(dev_ug_neighbors, hst_ug_neighbors_, numAgents*numNeighbors*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_num_neighbors, hst_num_neighbors_, numAgents*sizeof(int), cudaMemcpyHostToDevice);
#endif

	// Compute the unique neighbor counts
	cudaMemcpy(dev_neighbor_counts, dev_num_neighbors, sizeof(int)*numAgents, cudaMemcpyDeviceToDevice);
	thrust::device_ptr<int> thrust_neighbor_counts = thrust::device_pointer_cast(dev_neighbor_counts);
	thrust::sort(thrust::device,thrust_neighbor_counts, thrust_neighbor_counts+numAgents);

	cudaMalloc((void**)&dev_unique_indicators, numAgents*sizeof(bool));
	cudaMemset(dev_unique_indicators, false, numAgents*sizeof(bool));
	cudaMalloc((void**)&dev_unique_idxes, numAgents*sizeof(int));

	int* hst_neighbor_counts = (int*)malloc(numAgents*sizeof(int));
	cudaMemcpy(hst_neighbor_counts, dev_neighbor_counts, numAgents*sizeof(int), cudaMemcpyDeviceToHost);
	
	kernUnique<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_unique_indicators, dev_neighbor_counts);
	cudaThreadSynchronize();

	bool* hst_unique_indicators = (bool*)malloc(numAgents*sizeof(bool));
	cudaMemcpy(hst_unique_indicators, dev_unique_indicators,numAgents*sizeof(bool), cudaMemcpyDeviceToHost);

	int num_unique;
	bool last_unique_indicator;
	//thrust::device_ptr<int> thrust_unique_indicators = thrust::device_pointer_cast(dev_unique_indicators);
	thrust::exclusive_scan(thrust::device, dev_unique_indicators, dev_unique_indicators + numAgents, dev_unique_idxes);

	cudaMemcpy(&last_unique_indicator, dev_unique_indicators + numAgents - 1, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(&num_unique, dev_unique_idxes + numAgents - 1, sizeof(int), cudaMemcpyDeviceToHost);

	num_unique += last_unique_indicator;

	cudaMalloc((void**)&dev_unique_counts, numAgents*sizeof(int));
	kernGather<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_unique_counts, dev_neighbor_counts, dev_unique_indicators, dev_unique_idxes);

	int* hst_unique_counts = (int*)malloc(num_unique*sizeof(int));
	cudaMemcpy(hst_unique_counts, dev_unique_counts, num_unique*sizeof(int), cudaMemcpyDeviceToHost);

	// Iterate through the number of neighbors and update agents accordingly
	int num_sub_agents;
	bool last_indicator;

	for (int i = 0; i < num_unique; i++){

		int n = hst_unique_counts[i];

		if (n == 0) continue;

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

		//printf("---num_neighbors: %d, num_agent: %d---\n", n, num_sub_agents);

		cudaFree(dev_idx);
		cudaFree(dev_neighbor_ids);
		cudaFree(dev_agent_ids);
		cudaFree(dev_hrvos);
		cudaFree(dev_candidates);
		cudaFree(dev_vel_new);
		cudaFree(dev_in_pcr);
		
		numHRVOs = n;
		totHRVOs = num_sub_agents*n;
		int numNaiveCandidates = 2 * numHRVOs;
		int numMaxVelCandidates = 4 * numHRVOs;
		int numIntersectionCandidates = 4 * numHRVOs * (numHRVOs - 1);
		numCandidates = numNaiveCandidates + numMaxVelCandidates + numIntersectionCandidates;
		//numCandidates = numNaiveCandidates + numIntersectionCandidates;
		totCandidates = num_sub_agents * numCandidates;

		cudaMalloc((void**)&dev_idx, num_sub_agents*sizeof(int));
		cudaMalloc((void**)&dev_agent_ids, num_sub_agents*sizeof(int));
		cudaMalloc((void**)&dev_neighbor_ids, num_sub_agents*n*sizeof(int));
		cudaMalloc((void**)&dev_hrvos, totHRVOs*sizeof(HRVO));
		cudaMalloc((void**)&dev_candidates, totCandidates*sizeof(CandidateVel));
		cudaMalloc((void**)&dev_vel_new, num_sub_agents*sizeof(glm::vec3));
		cudaMalloc((void**)&dev_in_pcr, num_sub_agents*sizeof(bool));

		cudaMemset(dev_in_pcr, false, num_sub_agents*sizeof(bool));
		cudaThreadSynchronize();

		dim3 fullBlocksForNeighborIds((n*num_sub_agents + blockSize - 1) / blockSize);
		dim3 fullBlocksForAgentIds((num_sub_agents + blockSize - 1) / blockSize);
		dim3 fullBlocksForHRVOs((totHRVOs + blockSize - 1) / blockSize);
		dim3 fullBlocksForCandidates((totCandidates + blockSize - 1) / blockSize);

		//printf("Candidates: %d,  blocks: %d\n", totCandidates, fullBlocksForCandidates.x);

		printf("-----Num Neighbors: %d-----\n",n);

		kernGetSubsetIdxes<<<fullBlocksPerGrid,blockSize>>>(numAgents, dev_idx, dev_indicators, dev_indicator_sum, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc subset idxes failed!");

		kernGetSubsetAgentIds<<<fullBlocksForAgentIds, blockSize>>>(num_sub_agents, dev_agent_ids, dev_idx, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc subsetagentsid failed!");

		kernGetSubsetNeighborIds<<<fullBlocksForNeighborIds, blockSize>>>(num_sub_agents*n, n, numNeighbors, dev_neighbor_ids, dev_ug_neighbors, dev_idx);
		checkCUDAErrorWithLine("cudaMalloc subsetneighborsid failed!");

		cudaEvent_t startHRVO, stopHRVO;
		cudaEventCreate(&startHRVO);
		cudaEventCreate(&stopHRVO);
		cudaEventRecord(startHRVO);

		kernComputeHRVOs<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, numHRVOs, num_sub_agents, dev_hrvos, dev_agent_ids, dev_agents, dev_neighbor_ids, dt);
		checkCUDAErrorWithLine("cudaMalloc dev_goals failed!");

		cudaEventRecord(stopHRVO);
		cudaEventSynchronize(stopHRVO);
		float millisecondsHRVO = 0;
		cudaEventElapsedTime(&millisecondsHRVO, startHRVO, stopHRVO);
		printf("HRVO: %f\n", millisecondsHRVO);

		cudaEvent_t startCAN, stopCAN;
		cudaEventCreate(&startCAN);
		cudaEventCreate(&stopCAN);
		cudaEventRecord(startCAN);

		kernInitCandidateVels<<<fullBlocksForCandidates, blockSize>>>(totCandidates, dev_candidates);
		checkCUDAErrorWithLine("cudaMalloc cand vels failed!");

		kernComputeNaiveCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, num_sub_agents, numHRVOs, numCandidates, dev_candidates, dev_hrvos, dev_agent_ids, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc naive vels failed!");

		kernComputeMaxCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, num_sub_agents, numHRVOs, numCandidates, numNaiveCandidates, dev_candidates, dev_hrvos, dev_agent_ids, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc max vels failed!");

		kernComputeIntersectionCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, num_sub_agents, numHRVOs, numCandidates, numNaiveCandidates+numMaxVelCandidates, dev_candidates, dev_hrvos, dev_agent_ids, dev_agents);
		//kernComputeIntersectionCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, num_sub_agents, numHRVOs, numCandidates, numNaiveCandidates, dev_candidates, dev_hrvos, dev_agent_ids, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc intersection vels failed!");

		cudaEventRecord(stopCAN);
		cudaEventSynchronize(stopCAN);
		float millisecondsCAN = 0;
		cudaEventElapsedTime(&millisecondsCAN, startCAN, stopCAN);
		printf("Candidates: %f\n", millisecondsCAN);

		cudaEvent_t startVALID, stopVALID;
		cudaEventCreate(&startVALID);
		cudaEventCreate(&stopVALID);
		cudaEventRecord(startVALID);

		kernComputeValidCandidateVels<<<fullBlocksForCandidates, blockSize>>>(totCandidates, num_sub_agents, numHRVOs, numCandidates, dev_candidates, dev_hrvos, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc compute valid vels failed!");

		cudaEventRecord(stopVALID);
		cudaEventSynchronize(stopVALID);
		float millisecondsVALID = 0;
		cudaEventElapsedTime(&millisecondsVALID, startVALID, stopVALID);
		printf("Computing Valid: %f\n", millisecondsVALID);

		cudaEvent_t startBEST, stopBEST;
		cudaEventCreate(&startBEST);
		cudaEventCreate(&stopBEST);
		cudaEventRecord(startBEST);

		kernComputeBestVel<<<fullBlocksPerGrid, blockSize>>>(num_sub_agents, numCandidates, dev_vel_new, dev_candidates);
		checkCUDAErrorWithLine("cudaMalloc best vels failed!");

		cudaEventRecord(stopBEST);
		cudaEventSynchronize(stopBEST);
		float millisecondsBEST = 0;
		cudaEventElapsedTime(&millisecondsBEST, startBEST, stopBEST);
		printf("Selecting Best: %f\n", millisecondsBEST);

		kernComputeInPCR<<<fullBlocksPerGrid, blockSize>>>(num_sub_agents, numHRVOs, dev_in_pcr, dev_hrvos, dev_agent_ids, dev_agents);
		checkCUDAErrorWithLine("cudaMalloc in pcr failed!");

		kernUpdateVel<<<fullBlocksPerGrid, blockSize>>>(num_sub_agents, dev_agent_ids, dev_agents, dev_in_pcr, dev_vel_new);
		checkCUDAErrorWithLine("cudaMalloc update vels failed!");
	}

	// Update the positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_pos);



}
