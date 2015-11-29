#define GLM_FORCE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
//#include <vector>

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
#define robot_radius 0.5
#define circle_radius 4
#define desired_speed 3.0f

#define GRIDMAX 20 // Must be even, creates grid of GRIDMAX x GRIDMAX size
#define NNRADIUS 2.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numAgents;
// TODO: bottom few will need to get removed (# will vary in the loop)
int numFVOs; // total # of FVOs
int totFVOs;
int numNeighbors; // Neighbors per agent
int totNeighbors;
int numIntersections; // Nnumber of intersection points per agent
int totIntersections;
int numConstraints; // Number of constraints in the Boundary Edge set per agent
int totConstraints;

dim3 threadsPerBlock(blockSize);

float scene_scale = 1.0;

glm::vec2* dev_endpoints;
glm::vec3* dev_pos;
agent* dev_agents;
FVO* dev_fvos;
int* dev_neighbors; //index of neighbors
bool* dev_in_pcr;
ray* dev_rays;
constraint* dev_constraints;
UGEntry* dev_uglist;
int* dev_startIdx;

int* dev_ug_neighbors;
int* dev_num_neighbors;

glm::vec3* dev_closest_points;
intersection* dev_intersections;

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

// Computes the intersection between 2 rays
// point is the intersection, bool returns true if there is an intersection, false otherwise
__host__ __device__ intersection intersectRayRay(ray a, ray b){
	// Solves p1 + t1*v1 = p2 + t2*v2
	// [t1; t2] = [v1x -v2x; v1y -v2y]^-1*(p2-p1);

	intersection point;
	point.isIntersection = false;
	
	// Parallel lines cannot intersect
	if (a.dir.x == b.dir.x && a.dir.y == b.dir.y){
		return point;
	}

	glm::vec2 ts;

	ts = glm::inverse(glm::mat2(a.dir.x, a.dir.y, -b.dir.x, -b.dir.y)) * glm::vec2(b.pos - a.pos);

	if (ts.x >= 0 && ts.y >= 0){
		point.point = glm::vec3(a.pos+a.dir*ts.x);
		point.point.z = 0.0;
		point.isIntersection = true;
		return point;
	}
	return point;
}

__host__ __device__ intersection intersectSegmentSegment(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 p4){
	//https://www.topcoder.com/community/data-science/data-science-tutorials/geometry-concepts-line-intersection-and-its-applications/
	
	intersection point;
	point.isIntersection = false;

	float A1 = p2.y - p1.y;
	float B1 = p1.x - p2.x;
	float C1 = A1*p1.x + B1*p1.y;

	float A2 = p4.y - p3.y;
	float B2 = p3.x - p4.x;
	float C2 = A2*p3.x + B2*p3.y;

	float det = A1*B2 - A2*B1;
	if (det == 0){
		return point;
	}
	point.point.x = (B2*C1 - B1*C2) / det;
	point.point.y = (A1*C2 - A2*C1) / det;
	
	point.isIntersection = glm::min(p1.x, p2.x) <= point.point.x && 
						   glm::max(p1.x, p2.x) >= point.point.x && 
						   glm::min(p1.y, p2.y) <= point.point.y &&
						   glm::max(p1.y, p2.y) >= point.point.y &&
						   glm::min(p3.x, p4.x) <= point.point.x &&
						   glm::max(p3.x, p4.x) >= point.point.x &&
						   glm::min(p3.y, p4.y) <= point.point.y &&
						   glm::max(p3.y, p4.y) >= point.point.y;
	return point;
}

__host__ __device__ intersection intersectRay2Segment(ray a, glm::vec3 p1, glm::vec3 p2){
	intersection p = intersectSegmentSegment(a.pos, a.pos+a.dir, p1, p2);

	glm::vec3 pdir = glm::normalize(p.point - a.pos);

	//TODO check if parallel
	p.isIntersection = glm::min(p1.x, p2.x) <= p.point.x &&
					   glm::max(p1.x, p2.x) >= p.point.x &&
					   glm::min(p1.y, p2.y) <= p.point.y &&
					   glm::max(p1.y, p2.y) >= p.point.y &&
					   glm::dot(a.dir, pdir) > 0;
	return p;
}

__host__ __device__ intersection intersectRaySegment(ray a, glm::vec3 p1, glm::vec3 p2){
	// http://stackoverflow.com/questions/14307158/how-do-you-check-for-intersection-between-a-line-segment-and-a-line-ray-emanatin
	// http://gamedev.stackexchange.com/questions/85850/collision-intersection-of-2d-ray-to-line-segment
	// https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/

	intersection point;
	point.isIntersection = false;

	glm::vec3 v1 = a.pos - p1;
	glm::vec3 v2 = p2 - p1;
	glm::vec3 v3 = glm::vec3(-a.dir.y, a.dir.x, 0.0);

	float t1 = glm::length(glm::cross(v2,v1))/glm::dot(v2,v3);
	float t2 = glm::dot(v1, v3) / glm::dot(v2, v3);

	if (t1 >= 0 && t2 >= 0 && t2 <= 1)
	{
		// Return the point of intersection
		point.point = a.pos + t1 * a.dir;
		point.isIntersection = true;
	}

	return point;
}

__host__ __device__ glm::vec3 projectPointToLine(glm::vec3 a, glm::vec3 b, glm::vec3 p){
	// A + dot(AP,AB) / dot(AB,AB) * AB
	// http://gamedev.stackexchange.com/questions/72528/how-can-i-project-a-3d-point-onto-a-3d-line

	glm::vec3 ap = p - a;
	glm::vec3 ab = b - a;

	return a + glm::dot(ap, ab) / glm::dot(ab, ab) * ab;
}

__host__ __device__ glm::vec3 projectPointToRay(ray a, glm::vec3 p){
	// http://stackoverflow.com/questions/5227373/minimal-perpendicular-vector-between-a-point-and-a-line

	return (a.pos + (glm::normalize(p - a.pos))*a.dir);
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

__host__ __device__ void intersectFVOtoFVO(FVO a, FVO b, intersection* points){
	points[0] = intersectRayRay(a.L.ray, b.L.ray);
	points[1] = intersectRaySegment(a.L.ray, b.L.ray.pos, b.R.ray.pos);
	points[2] = intersectRayRay(a.L.ray, b.R.ray);
	
	points[3] = intersectRaySegment(b.L.ray, a.L.ray.pos, a.R.ray.pos);
	points[4] = intersectSegmentSegment(a.L.ray.pos, a.R.ray.pos, b.L.ray.pos, b.R.ray.pos);
	points[5] = intersectRaySegment(b.R.ray, a.L.ray.pos, a.R.ray.pos);
	
	points[6] = intersectRayRay(a.R.ray, b.L.ray);
	points[7] = intersectRaySegment(a.R.ray, b.L.ray.pos, b.R.ray.pos);
	points[8] = intersectRayRay(a.R.ray, b.R.ray);
}

__host__ __device__ FVO computeFVO(agent A, agent B){
	glm::vec3 pAB = B.pos - A.pos;
	float radius = A.radius + B.radius;

	FVO fvo;
	constraint T, L, R;

	// Compute pAB perpendicular
	// TODO: how do I figure out which direction this should be on?

	glm::vec3 pABp = glm::cross(glm::normalize(pAB), glm::vec3(0.0, 0.0, 1.0));

	// Compute FVO_T
	float sep = glm::length(pAB) - radius;
	float n = glm::tan(glm::asin(radius / glm::length(pAB)))*sep;
	glm::vec3 med_vel = (A.vel + B.vel) / 2.0f;

	glm::vec3 M = sep*glm::normalize(pAB) + med_vel;
	M += A.pos; //TODO: Do I need to multiple the vA + vB/2 by the time step?
	
	T.ray.pos = M - glm::normalize(pABp)*n;
	T.ray.dir = glm::normalize(pABp);
	T.norm = glm::normalize(pAB);

	// Compute FVO_L
	glm::vec3 apex = A.pos + (A.vel + B.vel) / 2.0f;
	float theta = glm::asin(radius / glm::length(pAB));

	glm::vec3 rotatedL = glm::rotateZ(pAB, theta);

	L.ray.pos = T.ray.pos;
	L.ray.dir = glm::normalize(rotatedL);
	ray pABl;
	pABl.pos = apex;
	pABl.dir = L.ray.dir;
	L.norm = glm::normalize(intersectPointRay(pABl, B.pos));

	// Compute FVO_R
	glm::vec3 rotatedR = glm::rotateZ(pAB, -theta);

	R.ray.pos = M + glm::normalize(pABp)*n;
	R.ray.dir = glm::normalize(rotatedR);
	ray pABr;
	pABr.pos = apex;
	pABr.dir = R.ray.dir;
	R.norm = glm::normalize(intersectPointRay(pABr, B.pos));

	L.isRay = true;
	T.endpoint = R.ray.pos;
	T.isRay = false;
	R.isRay = true;

	fvo.T = T;
	fvo.L = L;
	fvo.R = R;

	return fvo;
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

		//if (index == 1){
		//	rad -= 3.1415f / 2.0f;
		//}

		agents[index].pos.x = scale * circle_radius * cos(rad);
		agents[index].pos.y = scale * circle_radius * sin(rad);
		agents[index].pos.z = 0.0;

		/*
		if (index == 0){
			agents[index].pos = glm::vec3(1.0,1.0,0.0);
		}
		else if (index == 1){
			agents[index].pos = glm::vec3(1.0, 1.0, 0.0);
		}
		else if (index == 2){
			agents[index].pos = glm::vec3(1.0, 1.0, 0.0);
		}
		else if (index == 3){
			agents[index].pos = glm::vec3(-2.0, 2.0, 0.0);
		}
		*/

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

	cudaMalloc((void**)&dev_in_pcr, N*sizeof(bool));
	checkCUDAErrorWithLine("cudaMalloc dev_in_pcr failed!");

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
void ClearPath::copyAgentsToVBO(float *vbodptr, glm::vec2* endpoints, glm::vec3* pos, agent* agents, intersection* intersections, int* neighbors,  int* num_neighbors) {
    dim3 fullBlocksPerGrid((int)ceil(float(numAgents) / float(blockSize)));

    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_pos, vbodptr, scene_scale);
    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");

	//kernCopyFVOtoEndpoints<<<fullBlocksPerGrid, blockSize>>>(3*(numAgents-1), dev_endpoints);
	cudaMemcpy(endpoints, dev_endpoints, 6*(numFVOs)*sizeof(glm::vec2), cudaMemcpyDeviceToHost);

	cudaMemcpy(pos, dev_pos, numAgents*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(agents, dev_agents, numAgents*sizeof(agent), cudaMemcpyDeviceToHost);
	cudaMemcpy(intersections, dev_intersections, totIntersections*sizeof(intersection), cudaMemcpyDeviceToHost);
	
	// UG
	//cudaMemcpy(neighbors, dev_ug_neighbors, numAgents*numNeighbors*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(num_neighbors, dev_num_neighbors, numAgents*sizeof(int), cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
}

/******************
 * stepSimulation *
 ******************/

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

__global__ void kernComputeFVOs(int N, int numNeighbors, FVO* fvos, agent* agents, int* neighbors){
	// N = number of agents
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		//int numNeighbors = N - 1;
		for (int i = 0; i < numNeighbors; i++){
			fvos[i + numNeighbors*index] = computeFVO(agents[index], agents[neighbors[i + numNeighbors*index]]);
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

__global__ void kernCheckInPCR(int N, int numNeighbors, bool* in_pcr, agent* agents, FVO* fvos){
	// N - number of agents
	// TODO: increase utilization by putting into for loop and compacting
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < N){
		//int numNeighbors = N - 1;
		for (int i = 0; i < numNeighbors; i++){
			in_pcr[index] = pointInFVO(fvos[index*numNeighbors + i], agents[index].pos + agents[index].vel);
		}
	}

}

__global__ void kernUpdateVelBad(int N, float dt, agent *agents, FVO* fvos, bool* in_pcr){
	//TODO: can improve this by compacting beforehand?
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// THIS ONLY WORKS FOR 2 ROBOTS

	if (index < N && in_pcr[index]){
		glm::vec3 des_vel = agents[index].vel;

		glm::vec3 min_vel;

		// TODO: the 10.0f * dir is a hack...figure out how to project onto a ray instead
		glm::vec3 minl = fvos[index].L.ray.pos;
		glm::vec3 minr = fvos[index].R.ray.pos;
			
		if (glm::distance(minl, agents[index].pos+des_vel) < glm::distance(minr, agents[index].pos + des_vel)){
			min_vel = minl;
		}
		else {
			min_vel = minr;
		}

		agents[index].vel = min_vel-agents[index].pos;
	}
}

__global__ void kernUpdateVel2Only(int N, float dt, agent *agents, FVO* fvos, bool* in_pcr){
	//TODO: can improve this by compacting beforehand?
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// THIS ONLY WORKS FOR 2 ROBOTS

	if (index < N && in_pcr[index]){
		glm::vec3 des_vel = agents[index].vel;

		glm::vec3 min_vel;

		// TODO: the 10.0f * dir is a hack...figure out how to project onto a ray instead
		glm::vec3 minl = projectPointToLine(fvos[index].L.ray.pos, fvos[index].L.ray.pos+10.0f*fvos[index].L.ray.dir, agents[index].pos + des_vel) - agents[index].pos;
		glm::vec3 minr = projectPointToLine(fvos[index].R.ray.pos, fvos[index].R.ray.pos + 10.0f*fvos[index].R.ray.dir, agents[index].pos + des_vel) - agents[index].pos;
		glm::vec3 mint = projectPointToLine(fvos[index].L.ray.pos, fvos[index].R.ray.pos, agents[index].pos + des_vel) - agents[index].pos;

		if (glm::distance(minl, des_vel) < glm::distance(minr, des_vel) && glm::distance(minl, des_vel) < glm::distance(mint, des_vel)){
			min_vel = minl;
		}
		else if (glm::distance(mint, des_vel) < glm::distance(minr, des_vel) && glm::distance(mint, des_vel) < glm::distance(mint, des_vel)){
			min_vel = mint;
		}
		else {
			min_vel = minr;
		}

		agents[index].vel = min_vel;
	}
}

__global__ void kernConvertFVOsToConstraints(int numFVOs, int numAgents, int numNeighbors, constraint* constraints, FVO* fvos){
	//N is the number of FVOs
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numFVOs){
		int row_FVO = index / numNeighbors;
		int col_FVO = index % numNeighbors;
		int numConstraints = numNeighbors * 3;

		constraints[numConstraints*row_FVO + 3 * col_FVO] = fvos[index].L;
		constraints[numConstraints*row_FVO + 3 * col_FVO + 1] = fvos[index].T;
		constraints[numConstraints*row_FVO + 3 * col_FVO + 2] = fvos[index].R;
	}
}

__global__ void kernFindIntersections(int totConstraints, int numAgents, int numConstraints, int numIntersections, intersection* intersections, const constraint* constraints){
	// totConstraints is the number of constraints
	// Parallelizes over the constraints
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totConstraints){
		// Get current FVO
		int cr = index / numConstraints;
		int cc = index % numConstraints;

		intersection point;
		constraint c = constraints[index];
		constraint oc;

		// Iterate through all the other constraints
		int j = 0;

		// %3 in order to skip comparing an FVO with itself
		for (int i = 0; i < cc - (cc % 3); i++){
			oc = constraints[i + cr*numConstraints];

			if (c.isRay && oc.isRay){
				point = intersectRayRay(c.ray, oc.ray);
			}
			else if (c.isRay && !oc.isRay){
				point = intersectRay2Segment(c.ray,oc.ray.pos,oc.endpoint);
			}
			else if (!c.isRay && oc.isRay){
				point = intersectRay2Segment(oc.ray, c.ray.pos, c.endpoint);
			}
			else {
				point = intersectSegmentSegment(c.ray.pos, c.endpoint, oc.ray.pos, oc.endpoint);
			}

			/*
			point = intersectRayRay(c.ray,oc.ray);
			
			point.isIntersection = point.isIntersection && (
								   (c.isRay && oc.isRay) ||
								   (c.isRay && !oc.isRay && glm::distance(oc.ray.pos,point.point) <= glm::distance(oc.ray.pos, oc.endpoint)) ||
								   (!c.isRay && oc.isRay && glm::distance(c.ray.pos, point.point) <= glm::distance(c.ray.pos, c.endpoint)) ||
								   (!c.isRay && !oc.isRay && glm::distance(c.ray.pos, point.point) <= glm::distance(c.ray.pos, c.endpoint) && glm::distance(oc.ray.pos, point.point) <= glm::distance(oc.ray.pos, oc.endpoint)));
			*/

			intersections[cc*(numConstraints-3) + j + cr*numIntersections] = point;
			j++;
		}

		for (int i = cc + (3-cc%3); i < numConstraints; i++){
			oc = constraints[i + cr*numConstraints];

			if (c.isRay && oc.isRay){
				point = intersectRayRay(c.ray, oc.ray);
			}
			else if (c.isRay && !oc.isRay){
				point = intersectRay2Segment(c.ray, oc.ray.pos, oc.endpoint);
			}
			else if (!c.isRay && oc.isRay){
				point = intersectRay2Segment(oc.ray, c.ray.pos, c.endpoint);
			}
			else {
				point = intersectSegmentSegment(c.ray.pos, c.endpoint, oc.ray.pos, oc.endpoint);
			}

			/*
			point = intersectRayRay(c.ray, oc.ray);
			
			point.isIntersection = point.isIntersection && (
				(c.isRay && oc.isRay) ||
				(c.isRay && !oc.isRay && glm::distance(oc.ray.pos, point.point) <= glm::distance(oc.ray.pos, oc.endpoint)) ||
				(!c.isRay && oc.isRay && glm::distance(c.ray.pos, point.point) <= glm::distance(c.ray.pos, c.endpoint)) ||
				(!c.isRay && !oc.isRay && glm::distance(c.ray.pos, point.point) <= glm::distance(c.ray.pos, c.endpoint) && glm::distance(oc.ray.pos, point.point) <= glm::distance(oc.ray.pos, oc.endpoint)));
			*/

			intersections[cc*(numConstraints-3) + j + cr*numIntersections] = point;
			j++;
		}
	}
}

__global__ void kernUpdatePos(int N, float dt, agent* dev_agents, glm::vec3 *dev_pos){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N){
		// Update positions
		dev_agents[index].pos = dev_agents[index].pos + dev_agents[index].vel * dt;
		dev_pos[index] = dev_agents[index].pos;
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


/**
 * Step the entire N-body simulation by `dt` seconds.
 */
void ClearPath::stepSimulation(float dt) {

	glm::vec3 p1 = glm::vec3(0.0);
	glm::vec3 p2 = glm::vec3(0.0,1.0,0.0);
	glm::vec3 p3 = glm::vec3(0.5,0.5,0.0);
	glm::vec3 p4 = glm::vec3(1.0,0.5,0.0);

	intersection p = intersectSegmentSegment(p1,p2,p3,p4);

	printf("%d: %f %f\n", p.isIntersection, p.point.x, p.point.y);

	dim3 fullBlocksPerGrid((numAgents + blockSize - 1) / blockSize);

	// Update all the desired velocities given current positions
	kernUpdateDesVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_agents);

	// Allocate space for neighbors, FVOs, intersection points (how to do this?)
	numNeighbors = numAgents - 1;
	numFVOs = numNeighbors;
	numConstraints = numFVOs * 3; // number of constraints that each robot has to compare
	numIntersections = numConstraints * (numConstraints - 3); //-3 to remove constraints from the same FVO
	
	totNeighbors = numNeighbors * numAgents;
	totFVOs = numFVOs * numAgents;
	totConstraints = numConstraints * numAgents;
	totIntersections = numIntersections * numAgents;

	printf("num constraints per robot: %d\n", numConstraints);
	printf("total num constraints: %d\n", totConstraints);
	printf("num intersections per robot: %d\n", numIntersections);
	printf("dev intersections: %d\n",totIntersections);

	// Get the number of blocks we need
	dim3 fullBlocksForFVOs((totFVOs + blockSize - 1) / blockSize);
	dim3 fullBlocksForConstraints((totConstraints + blockSize - 1) / blockSize);
	dim3 fullBlocksForUG((GRIDMAX*GRIDMAX + blockSize - 1) / blockSize);

	// Free everything
	cudaFree(dev_neighbors);
	cudaFree(dev_fvos);
	cudaFree(dev_endpoints);
	
	cudaFree(dev_intersections);
	cudaFree(dev_constraints);

	// UG
	/*
	cudaFree(dev_uglist);
	cudaFree(dev_startIdx);
	cudaFree(dev_ug_neighbors);
	cudaFree(dev_num_neighbors);
	*/

	cudaMalloc((void**)&dev_neighbors, totFVOs*sizeof(int));
	cudaMalloc((void**)&dev_fvos, totFVOs*sizeof(FVO));
	cudaMalloc((void**)&dev_endpoints, 6*totFVOs*sizeof(glm::vec2));

	cudaMalloc((void**)&dev_constraints, totConstraints*sizeof(constraint));
	cudaMalloc((void**)&dev_intersections, totIntersections*sizeof(intersection));
	
	// UG
	/*
	cudaMalloc((void**)&dev_uglist, numAgents*sizeof(UGEntry));
	cudaMalloc((void**)&dev_startIdx, GRIDMAX*GRIDMAX*sizeof(int));
	cudaMalloc((void**)&dev_ug_neighbors, numNeighbors*numAgents*sizeof(int));
	cudaMalloc((void**)&dev_num_neighbors, numAgents*sizeof(int));
	*/

	// TODO: this should actually be a kernel initialization
	cudaMemset(dev_in_pcr, false, numAgents*sizeof(bool));
	cudaThreadSynchronize();

	// Find neighbors
	// TODO: 2 arrays: 1 of all the ones that have same # neighbors, other has the remaining uncomputed ones
	kernComputeNeighbors<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_neighbors, dev_agents);
	cudaThreadSynchronize();

	// Compute Neighbors with the Uniform Grid (UG)
	/*
	kernUpdateUniformGrid<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_uglist, dev_agents);
	//cudaThreadSynchronize();

	// Sort by cell id
	thrust::device_ptr<UGEntry> thrust_uglist = thrust::device_pointer_cast(dev_uglist);
	thrust::sort(thrust_uglist, thrust_uglist+numAgents, UGComp());

	// Create start index structure
	kernInitStartIdxes<<<fullBlocksForUG, blockSize>>>(GRIDMAX*GRIDMAX, dev_startIdx);
	kernUGStartIdxes<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_startIdx, dev_uglist);
	
	UGEntry* hst_uglist = (UGEntry*)malloc(numAgents * sizeof(UGEntry));
	cudaMemcpy(hst_uglist, dev_uglist, numAgents*sizeof(UGEntry), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numAgents; i++){
		printf("(cell %d, agent %d)\n", hst_uglist[i].cellId, hst_uglist[i].agentId);
	}

	printf("-----\n");
	int* hst_startIdx = (int*)malloc(GRIDMAX*GRIDMAX*sizeof(int));
	cudaMemcpy(hst_startIdx, dev_startIdx, GRIDMAX*GRIDMAX*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < GRIDMAX*GRIDMAX; i++){
		if (hst_startIdx[i] >= 0){
			printf("cell: %d, start: %d\n", i, hst_startIdx[i]);
		}
	}

	free(hst_uglist);
	free(hst_startIdx);

	// Get the neighbors

	kernComputeUGNeighbors<<<fullBlocksPerGrid, blockSize>>>(numAgents, numNeighbors, dev_ug_neighbors, dev_num_neighbors, dev_agents, dev_startIdx, dev_uglist);

	int* hst_ug_neighbors = (int*)malloc(numAgents*numNeighbors*sizeof(int));
	int* hst_num_neighbors = (int*)malloc(numAgents*sizeof(int));

	cudaMemcpy(hst_ug_neighbors, dev_ug_neighbors, numAgents*numNeighbors*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_num_neighbors, dev_num_neighbors, numAgents*sizeof(int), cudaMemcpyDeviceToHost);


	for (int i = 0; i < numAgents; i++){
		printf("agent %d: ", i);
		for (int j = 0; j < hst_num_neighbors[i]; j++){
			printf("%d ", hst_ug_neighbors[i*numNeighbors+j]);
		}
		printf("\n");
	}

	free(hst_ug_neighbors);
	free(hst_num_neighbors);
	*/

	// Compute the FVOs
	kernComputeFVOs<<<fullBlocksPerGrid, blockSize>>>(numAgents, numNeighbors, dev_fvos, dev_agents, dev_neighbors);

	//TODO: change FullBlocksPerGrid here to fullblockforfvo?
	kernCopyFVOtoEndpoints<<<fullBlocksPerGrid, blockSize>>>(numFVOs, dev_endpoints, dev_fvos);

	// See if velocity is in PCR, if so, continue, else skip and just update the velocity
	kernCheckInPCR<<<fullBlocksPerGrid, blockSize>>>(numAgents, numNeighbors, dev_in_pcr, dev_agents, dev_fvos);

	// Gather each constraint so we can easily track intersections
	kernConvertFVOsToConstraints<<<fullBlocksForFVOs, blockSize>>>(totFVOs, numAgents, numNeighbors, dev_constraints, dev_fvos);

	// Compute all the intersection points
	kernFindIntersections<<<fullBlocksForConstraints, blockSize>>>(totConstraints, numAgents, numConstraints, numIntersections, dev_intersections, dev_constraints);

	intersection* hst_intersection = (intersection*)malloc(totIntersections*sizeof(intersection));
	cudaMemcpy(hst_intersection, dev_intersections, totIntersections*sizeof(intersection), cudaMemcpyDeviceToHost);

	for (int i = 0; i < numIntersections; i++){
		if (hst_intersection[i].isIntersection){
			printf("%d  ",i);
		}
	}


	// Compute Inside/Outside Points

	// Sort Intersection Points based on distance from endpoint

	// Compute Inside/Outside Line Segments, track the nearest somehow

	// Update the velocities according to ClearPath
	//kernUpdateVel2Only<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_fvos, dev_in_pcr);
	//kernUpdateVelBad << <fullBlocksPerGrid, blockSize >> >(numAgents, dt, dev_agents, dev_fvos, dev_in_pcr);

	// Update the positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_pos);


}
