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
#define robot_radius 1
#define circle_radius 5


/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numAgents;
// TODO: bottom few will need to get removed (# will vary in the loop)
int numFVOs; // total # of FVOs
int numNeighbors; // Neighbors per agent
int numIntersectionPoints;

dim3 threadsPerBlock(blockSize);

float scene_scale = 1.0;

glm::vec2* dev_endpoints;
glm::vec3* dev_pos;
agent* dev_agents;
FVO* dev_fvos;
int* dev_neighbors; //index of neighbors

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

__host__ __device__ void computeFVO(agent A, agent B, FVO& fvo_out){
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

	fvo_out.T = T;
	fvo_out.L = L;
	fvo_out.R = R;

	//fvo_out = fvo;
	//return fvo;
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

__global__ void kernInitAgents(int N, agent* agents, float scale, float radius){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		float rad = ((float)index / (float)N) * (2.0f * 3.1415f);
		agents[index].pos.x = scale * circle_radius * cos(rad);
		agents[index].pos.y = scale * circle_radius * sin(rad);
		agents[index].pos.z = 0.0;
		agents[index].goal = -agents[index].pos;
		agents[index].radius = radius;
		agents[index].id = index;
	}
}

/**
 * Initialize memory, update some globals
 */
void ClearPath::initSimulation(int N) {
	//N = 5;
	numAgents = N;
    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_agents, N*sizeof(agent));
	checkCUDAErrorWithLine("cudaMalloc dev_goals failed!");

	// TODO: how do do only n-1 endpoints...?
	//cudaMalloc((void**)&dev_endpoints, 6*(N)*sizeof(glm::vec2));
	//checkCUDAErrorWithLine("cudaMalloc dev_endpoints failed!");

	//cudaMalloc((void**)&dev_fvos, (N)*sizeof(FVO));
	//checkCUDAErrorWithLine("cudaMalloc dev_fvos failed!");

	kernInitAgents<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents, scene_scale, robot_radius);
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

__global__ void kernCopyFVOtoEndpoints(int N, glm::vec2* endpoints, FVO* fvos){
	// N is the number of FVOs we have (number of endpoints / 2)
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < N){
		endpoints[6 * index] = glm::vec2(fvos[index].L.ray.pos);
		endpoints[6 * index + 1] = glm::vec2(fvos[index].L.ray.pos + 3.0f*fvos[index].L.ray.dir);
		endpoints[6 * index + 2] = glm::vec2(fvos[index].T.ray.pos);
		endpoints[6 * index + 3] = glm::vec2(fvos[index].R.ray.pos);
		endpoints[6 * index + 4] = glm::vec2(fvos[index].R.ray.pos);
		endpoints[6 * index + 5] = glm::vec2(fvos[index].R.ray.pos + 3.0f*fvos[index].R.ray.dir);
	}
}

/**
 * Wrapper for call to the kernCopyPlanetsToVBO CUDA kernel.
 */
void ClearPath::copyAgentsToVBO(float *vbodptr, glm::vec2* endpoints, glm::vec3* pos) {
    dim3 fullBlocksPerGrid((int)ceil(float(numAgents) / float(blockSize)));

    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_pos, vbodptr, scene_scale);
    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");

	//kernCopyFVOtoEndpoints<<<fullBlocksPerGrid, blockSize>>>(3*(numAgents-1), dev_endpoints);
	cudaMemcpy(endpoints, dev_endpoints, 6*(numFVOs)*sizeof(glm::vec2), cudaMemcpyDeviceToHost);

	cudaMemcpy(pos, dev_pos, numAgents*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

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
		dev_agents[index].vel = glm::normalize(dev_agents[index].goal - dev_agents[index].pos)*0.1f;
		float dist = glm::distance(dev_agents[index].goal, dev_agents[index].pos);
		if (dist < 0.1){
			dev_agents[index].vel = glm::vec3(0.0);
		}
	}
}

__global__ void kernComputeFVOs(int N, FVO* fvos, agent* agents, int* neighbors){
	// N = number of agents
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		int numNeighbors = N - 1;
		for (int i = 0; i < numNeighbors; i++){
			computeFVO(agents[index], agents[neighbors[i + numNeighbors*index]], fvos[i + numNeighbors*index]);
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
			neighbors[i-1 + index*numNeighbors] = i;
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
void ClearPath::stepSimulation(float dt) {

	dim3 fullBlocksPerGrid((numAgents + blockSize - 1) / blockSize);

	// Update all the desired velocities given current positions
	kernUpdateDesVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents);

	// Allocate space for neighbors, FVOs, intersection points (how to do this?)
	numNeighbors = numAgents - 1; // number of neighbors PER agent!
	numFVOs = numNeighbors * numAgents;
	numIntersectionPoints = numFVOs * numFVOs;

	// Free everything
	// TODO: Can we make any of these allocations static?
	cudaFree(dev_neighbors);
	cudaFree(dev_fvos);
	cudaFree(dev_endpoints);

	cudaMalloc((void**)&dev_neighbors, numFVOs*sizeof(int));
	cudaMalloc((void**)&dev_fvos, numFVOs*sizeof(FVO));
	cudaMalloc((void**)&dev_endpoints, 6*numFVOs*sizeof(glm::vec2));

	// Find neighbors
	// TODO: starting with all robots are neighbors
	// TODO: 2 arrays: 1 of all the ones that have same # neighbors, other has the remaining uncomputed ones
	kernComputeNeighbors<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_neighbors, dev_agents);

	int* hst_neighbors = (int*)malloc((numFVOs*numAgents));
	cudaMemcpy(hst_neighbors, dev_neighbors, numFVOs*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numFVOs; i++){
		printf("%d\t", hst_neighbors[i]);
	}
	printf("\n");
	//free(hst_neighbors);

	// Compute the FVOs
	kernComputeFVOs<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_fvos, dev_agents, dev_neighbors);

	//FVO* hst_fvos = (FVO*)malloc((numFVOs)*sizeof(FVO));
	//cudaMemcpy(hst_fvos, dev_fvos, (numFVOs)*sizeof(FVO), cudaMemcpyDeviceToHost);
	//free(hst_fvos);

	kernCopyFVOtoEndpoints<<<fullBlocksPerGrid, blockSize>>>(numFVOs, dev_endpoints, dev_fvos);

	//glm::vec2* hst_endpointss = (glm::vec2*)malloc(numFVOs*6*sizeof(glm::vec2));
	//cudaMemcpy(hst_endpointss, dev_endpoints, sizeof(glm::vec2)*numFVOs*6, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 6 * numFVOs; i++){
	//	printf("%f, %f\n", hst_endpointss[i].x, hst_endpointss[i].y);
	//}
	//printf("---");

	// See if velocity is in PCR, if so, continue, else skip and just update the velocity

	// Compute Intersection Points

	// Compute Inside/Outside Points

	// Sort Intersection Points based on distance from endpoint

	// Compute Inside/Outside Line Segments, track the nearest somehow

	// Update the velocities according to ClearPath
	kernUpdateVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents);

	// Update the positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_pos);


}
