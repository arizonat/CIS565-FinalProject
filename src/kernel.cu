#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define sign(x) (x>0)-(x<0)

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


/*******************
* Helper classes   *
********************/
struct agent{
	glm::vec3 pos;
	glm::vec3 vel;
	glm::vec3 goal;
	float radius;
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

/*******************
* Helper functions *
********************/

// Computes the intersection between 2 rays
// point is the intersection, bool returns true if there is an intersection, false otherwise
__host__ __device__ bool intersectRayRay(ray a, ray b, glm::vec3 &point){
	// Solves p1 + t1*v1 = p2 + t2*v2
	// [t1; t2] = [v1x -v2x; v1y -v2y]^-1*(p2-p1);
	
	// Parallel lines cannot intersect
	if (a.dir.x == b.dir.x && a.dir.y == b.dir.y){
		return false;
	}

	glm::vec2 ts;

	ts = glm::inverse(glm::mat2(a.dir.x, a.dir.y, -b.dir.x, -b.dir.y)) * glm::vec2(b.pos - a.pos);

	if (ts.x >= 0 && ts.y >= 0){
		point = glm::vec3(a.pos+a.dir*ts.x);
		point.z = 0.0;
		return true;
	}
	return false;
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

__host__ __device__ bool pointInFVO(FVO fvo, glm::vec3 p){
	// True if point is inside FVO, false otherwise (on border means it is NOT inside the FVO)
	// Basically check to see if all constraints are satisfied

	int fvol_sat = sidePointSegment(fvo.L.ray, p); // should be negative
	int fvor_sat = sidePointSegment(fvo.R.ray, p); // should be positive
	int fvot_sat = sidePointSegment(fvo.T.ray, p); // should be positive

	return (fvol_sat < 0) && (fvor_sat > 0) && (fvot_sat > 0);
}

__host__ __device__ FVO computeFVO(agent A, agent B){
	glm::vec3 pAB = B.pos - A.pos;
	float radius = A.radius + B.radius;

	FVO fvo;
	constraint T, L, R;

	// Compute pAB perpendicular
	// TODO: how do I figure out which direction this should be on?
	glm::vec3 pABp = glm::vec3(0.0);
	pABp.x = -pAB.y;
	pABp.y = -pAB.x;

	// Compute FVO_T
	float sep = glm::length(pAB) - radius;
	float n = glm::tan(glm::asin(radius / glm::length(pAB)))*sep;
	glm::vec3 M = sep*glm::normalize(pAB) + ((A.vel+B.vel)/2.0f);
	
	T.ray.pos = M - glm::normalize(pABp)*n;
	T.ray.dir = glm::normalize(pABp);
	T.norm = glm::normalize(pAB);

	fvo.T = T;

	// Compute FVO_L
	glm::vec3 apex = A.pos + (A.vel + B.vel) / 2.0f;
	float theta = glm::asin(radius / glm::length(pAB));

	glm::vec3 rotatedL = glm::rotate(pAB, theta, glm::vec3(0.0,0.0,1.0));

	L.ray.pos = T.ray.pos;
	L.ray.dir = glm::normalize(rotatedL);
	ray pABl;
	pABl.pos = apex;
	pABl.dir = L.ray.dir;
	L.norm = glm::normalize(intersectPointRay(pABl, B.pos));

	// Compute FVO_R
	glm::vec3 rotatedR = glm::rotate(pAB, -theta, glm::vec3(0.0,0.0,1.0));

	R.ray.pos = M + glm::normalize(pABp)*n;
	R.ray.dir = glm::normalize(rotatedR);
	ray pABr;
	pABr.pos = apex;
	pABr.dir = R.ray.dir;
	R.norm = glm::normalize(intersectPointRay(pABr, B.pos));

	fvo.T = T;
	fvo.L = L;
	fvo.R = R;

	return fvo;
}

/*****************
 * Configuration *
 *****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

#define robotRadius 1

/***********************************************
 * Kernel state (pointers are device pointers) *
 ***********************************************/

int numAgents;
dim3 threadsPerBlock(blockSize);

float scene_scale = 100.0;

glm::vec3 *dev_pos;
agent *dev_agents;

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

__global__ void kernInitAgents(int N, agent* agents, float scale, float radius){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		float rad = ((float)index / (float)N) * (2.0f * 3.1415f);
		agents[index].pos.x = scale * radius * cos(rad);
		agents[index].pos.y = scale * radius * sin(rad);
		agents[index].pos.z = 0.0;
		agents[index].goal = -agents[index].pos;
	}
}

/**
 * Initialize memory, update some globals
 */
void Nbody::initSimulation(int N) {
	//N = 5;
	numAgents = N;
    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_agents, N*sizeof(agent));
	checkCUDAErrorWithLine("cudaMalloc dev_goals failed!");

	kernInitAgents<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents, scene_scale, 0.5);
	checkCUDAErrorWithLine("kernInitAgents failed!");

    cudaThreadSynchronize();
}

/******************
 * copyPlanetsToVBO *
 ******************/

/**
 * Copy the planet positions into the VBO so that they can be drawn by OpenGL.
 */
__global__ void kernCopyPlanetsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale = -1.0f / s_scale;
	//float c_scale = 1.0f;

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
void Nbody::copyPlanetsToVBO(float *vbodptr) {
    dim3 fullBlocksPerGrid((int)ceil(float(numAgents) / float(blockSize)));

    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_pos, vbodptr, scene_scale);
    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");

    cudaThreadSynchronize();
}

/******************
 * stepSimulation *
 ******************/

__device__ glm::vec3 clearPathVelocity(){
	return glm::vec3();
}

__global__ void kernUpdateDesVel(int N, agent *dev_agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		// Get desired velocity
		dev_agents[index].vel = glm::normalize(dev_agents[index].goal - dev_agents[index].pos);
		if (glm::distance(dev_agents[index].goal, dev_agents[index].pos) < 0.1){
			dev_agents[index].vel = glm::vec3(0.0);
		}
	}
}

__global__ void kernUpdateVel(int N, float dt, agent *dev_agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		// Get ClearPath adjusted velocity


	}
}

__global__ void kernUpdatePos(int N, float dt, agent *dev_agents, glm::vec3 *dev_pos){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		// Update positions
		dev_agents[index].pos = dev_agents[index].pos + dev_agents[index].vel * dt;
		dev_pos[index] = dev_agents[index].pos;
	}
}

/**
 * Step the entire N-body simulation by `dt` seconds.
 */
void Nbody::stepSimulation(float dt) {

	dim3 fullBlocksPerGrid((numAgents + blockSize - 1) / blockSize);

	// Update all the desired velocities
	kernUpdateDesVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents);

	// Update the velocities according to ClearPath
	kernUpdateVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents);

	// Update the positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_pos);
}
