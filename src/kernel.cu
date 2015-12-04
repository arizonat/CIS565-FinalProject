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
#define circle_radius 9
#define desired_speed 3.0f

#define GRIDMAX 20 // Must be even, creates grid of GRIDMAX x GRIDMAX size
#define NNRADIUS 2.0f

// Experimental sampling
#define NUM_SAMPLES 250
#define MAX_VEL 3.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numAgents;
// TODO: bottom few will need to get removed (# will vary in the loop)
int numFVOs; // total # of FVOs
int totFVOs;
int numNeighbors; // Neighbors per agent
int totNeighbors;
int numIntersections; // Number of intersection points per agent
int totIntersections;
int numConstraints; // Number of constraints in the Boundary Edge set per agent
int totConstraints;

int totSamples;

int numHRVOs;
int totHRVOs;
int numCandidates;
int totCandidates;

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

int* dev_min_vel_diff;
glm::vec3* dev_vel_new;

int* dev_ug_neighbors;
int* dev_num_neighbors;

glm::vec3* dev_closest_points;
intersection* dev_intersections;

HRVO* dev_hrvos;
CandidateVel* dev_candidates;

// Experimental sampling
glm::vec3* dev_sample_vels;
float* dev_scores;

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

__host__ __device__ intersection intersectRaySegment(ray a, glm::vec3 p1, glm::vec3 p2){
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

__host__ __device__ bool pointInFVO(FVO fvo, glm::vec3 p){
	// True if point is inside FVO, false otherwise (on border means it is NOT inside the FVO)
	// Basically check to see if all constraints are satisfied

	// Allow some wiggle room?
	glm::vec3 pl = p - (fvo.L.norm)*0.0001f;
	glm::vec3 pr = p - (fvo.R.norm)*0.0001f;
	glm::vec3 pt = p - (fvo.T.norm)*0.0001f;

	int fvol_sat = sidePointSegment(fvo.L.ray, pl); // should be negative
	int fvor_sat = sidePointSegment(fvo.R.ray, pr); // should be positive
	int fvot_sat = sidePointSegment(fvo.T.ray, pt); // should be positive

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
	//ray pABl;
	//pABl.pos = apex;
	//pABl.dir = L.ray.dir;
	//L.norm = glm::normalize(intersectPointRay(pABl, B.pos));
	L.norm = glm::normalize(glm::cross(L.ray.dir, glm::vec3(0.0,0.0,1.0)));

	// Compute FVO_R
	glm::vec3 rotatedR = glm::rotateZ(pAB, -theta);

	R.ray.pos = M + glm::normalize(pABp)*n;
	R.ray.dir = glm::normalize(rotatedR);
	//ray pABr;
	//pABr.pos = apex;
	//pABr.dir = R.ray.dir;
	//R.norm = glm::normalize(intersectPointRay(pABr, B.pos));
	R.norm = glm::normalize(glm::cross(R.ray.dir, glm::vec3(0.0, 0.0, -1.0)));

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

		if (index == 1){
			rad -= 7.0*3.1415f / 8.0f;
		}

		//agents[index].w = 1.0f / float(index);
		agents[index].w = 1.0f;
		agents[index].pos.x = scale * circle_radius * cos(rad);
		agents[index].pos.y = scale * circle_radius * sin(rad);
		agents[index].pos.z = 0.0;

		agents[index].goal = -agents[index].pos;
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
void ClearPath::copyAgentsToVBO(float *vbodptr, glm::vec2* endpoints, glm::vec3* pos, agent* agents, HRVO* hrvos, CandidateVel* candidates, intersection* intersections, int* neighbors,  int* num_neighbors) {
    dim3 fullBlocksPerGrid((int)ceil(float(numAgents) / float(blockSize)));

    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_pos, vbodptr, scene_scale);
    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");

	//kernCopyFVOtoEndpoints<<<fullBlocksPerGrid, blockSize>>>(3*(numAgents-1), dev_endpoints);
	cudaMemcpy(endpoints, dev_endpoints, 6*(numFVOs)*sizeof(glm::vec2), cudaMemcpyDeviceToHost);

	cudaMemcpy(pos, dev_pos, numAgents*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(agents, dev_agents, numAgents*sizeof(agent), cudaMemcpyDeviceToHost);
	cudaMemcpy(intersections, dev_intersections, totIntersections*sizeof(intersection), cudaMemcpyDeviceToHost);
	
	// UG
	cudaMemcpy(neighbors, dev_ug_neighbors, numAgents*numNeighbors*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(num_neighbors, dev_num_neighbors, numAgents*sizeof(int), cudaMemcpyDeviceToHost);

	// HRVOs
	cudaMemcpy(hrvos, dev_hrvos, totHRVOs*sizeof(HRVO), cudaMemcpyDeviceToHost);
	cudaMemcpy(candidates, dev_candidates, totCandidates*sizeof(CandidateVel), cudaMemcpyDeviceToHost);

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

__global__ void kernUpdateVel(int numAgents, agent* agents, bool* in_pcr, glm::vec3* vel_new){
	int index = (blockDim.x*blockIdx.x) + threadIdx.x;

	if (index < numAgents){
		if (in_pcr[index]){
			agents[index].vel = vel_new[index];
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

__global__ void kernComputeHRVOs(int totHRVOs, int numHRVOs, int numAgents, HRVO* hrvos, agent* agents, int* neighbors, float dt){
	// Get all RVOs in parallel
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;
		hrvos[r*numHRVOs + c] = computeHRVO(agents[r], agents[neighbors[r*numHRVOs + c]], dt);

		//for (int i = 0; i < numNeighbors; i++){
		//	hrvos[i + numNeighbors*index] = computeHRVO(agents[index], agents[neighbors[i + numNeighbors*index]], dt);
		//}
	}
}

__global__ void kernInitCandidateVels(int totCandidates, CandidateVel* candidates){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totCandidates){
		candidates[index].valid = false;
	}
}

__global__ void kernComputeNaiveCandidateVels(int totHRVOs, int numAgents, int numHRVOs, int numCandidates, CandidateVel* candidates, HRVO* hrvos, agent* agents){
	// naive projection velocities
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;
		
		glm::vec3 prefVel = agents[r].vel;

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

__global__ void kernComputeIntersectionCandidateVels(int totHRVOs, int numAgents, int numHRVOs, int numCandidates, int offset, CandidateVel* candidates, HRVO* hrvos, agent* agents){
	// offset is usually the number of naiveCandidates (offset of starting location in the candidate buffer
	// intersection projection velocities
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totHRVOs){
		int r = index / numHRVOs;
		int c = index % numHRVOs;

		glm::vec3 prefVel = agents[r].vel;

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

__global__ void kernComputeInPCR(int numAgents, int numHRVOs, bool* in_pcr, HRVO* hrvos, agent* agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numAgents){
		for (int i = 0; i < numHRVOs; i++){
			if (det2(hrvos[index*numHRVOs + i].left, agents[index].vel - hrvos[index*numHRVOs + i].apex) < 0.0f &&
				det2(hrvos[index*numHRVOs + i].right, agents[index].vel - hrvos[index*numHRVOs + i].apex) > 0.0f){
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

/******************
* Sample-based RVO Kernels *
******************/

__global__ void kernInitMinVelDiff(int numAgents, int* min_vel_diff){
	int index = (blockDim.x*blockIdx.x) + threadIdx.x;

	if (index < numAgents){
		min_vel_diff[index] = INT_MAX;
	}
}

__global__ void kernSampleVelocities(int totSamples, float time, glm::vec3* sample_vels){
	// totSamples = numAgents * samplesPerAgent
	int index = (blockDim.x*blockIdx.x) + threadIdx.x;

	if (index < totSamples){
		sample_vels[index] = sampleRandom2DVelocity(time, index);
	}
}

__global__ void kernScoreSamples(int totSamples, int numAgents, int samplesPerAgent, float* scores, glm::vec3* sample_vels, agent* agents){
	int index = (blockDim.x*blockIdx.x) + threadIdx.x;

	if (index < totSamples){
		int sr = index / samplesPerAgent;
		int sc = index % samplesPerAgent;

		float wi = 1.0f;
		wi = agents[sr].w;

		float tc = INFINITY;
		float tci;

		glm::vec3 pA, pB, vA, vB, vABp;
		float t;
		bool intersects;
		float radius2;

		// Get time to collision
		for (int i = 0; i < numAgents; i++){
			if (i == sr) continue;

			pA = agents[sr].pos;
			vA = agents[sr].vel;
			vB = agents[i].vel;
			pB = agents[i].pos;
			vABp = 2.0f*sample_vels[index] - vA - vB;
			radius2 = (agents[i].radius + agents[sr].radius)*(agents[i].radius + agents[sr].radius);
			//intersects = glm::intersectRaySphere(pA, glm::normalize(vABp), pB, radius2, t);
			intersects = intersectRaySphere(pA, glm::normalize(vABp), pB, radius2, t);

			if (intersects && t < tc){
				tc = t;
			}
		}

		// Compute score
		if (tc != INFINITY){
			scores[index] = wi * (1.0 / tc) + glm::length(agents[sr].vel - sample_vels[index]);
		}
		else {
			scores[index] = glm::length(agents[sr].vel - sample_vels[index]);
		}
	}
}

__global__ void kernSelectVel(int numAgents, int samplesPerAgent, float* scores, glm::vec3* sample_vels, agent* agents, bool* in_pcr){
	int index = (blockDim.x*blockIdx.x) + threadIdx.x;

	if (index < numAgents && in_pcr[index]){
		float score = INFINITY;
		for (int i = 0; i < samplesPerAgent; i++){
			if (scores[i + index*samplesPerAgent] < score){
				score = scores[i + index*samplesPerAgent];
				agents[index].vel = sample_vels[i + index*samplesPerAgent];
			}
		}
	}
}

/******************
* FVO Kernels *
******************/

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

__global__ void kernCheckInPCR(int N, int numNeighbors, bool* in_pcr, agent* agents, FVO* fvos){
	// N - number of agents
	// TODO: increase utilization by putting into for loop and compacting
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (index < N){
		//int numNeighbors = N - 1;
		for (int i = 0; i < numNeighbors; i++){
			in_pcr[index] = pointInFVO(fvos[index*numNeighbors + i], agents[index].pos + agents[index].vel);
			in_pcr[index] = true;
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

		if (glm::distance(minl, agents[index].pos + des_vel) < glm::distance(minr, agents[index].pos + des_vel)){
			min_vel = minl;
		}
		else {
			min_vel = minr;
		}

		agents[index].vel = min_vel - agents[index].pos;
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
		glm::vec3 minl = projectPointToLine(fvos[index].L.ray.pos, fvos[index].L.ray.pos + 10.0f*fvos[index].L.ray.dir, agents[index].pos + des_vel) - agents[index].pos;
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

		// Iterate through all the other constraints and add intersections
		int j = 0;

		// My own startpoint is an intersection of this constraint
		point.point = c.ray.pos;
		point.isIntersection = true;
		point.distToOrigin = 0.0f;
		intersections[cc*(numConstraints - 1) + j + cr*numIntersections] = point;
		j++;

		// If T constraint, then endpoint is also an intersection
		if (!c.isRay){
			point.point = c.endpoint;
			point.isIntersection = true;
			point.distToOrigin = glm::distance(point.point, c.ray.pos);
			intersections[cc*(numConstraints - 1) + j + cr*numIntersections] = point;
			j++;
		}

		// %3 in order to skip comparing an FVO with itself
		for (int i = 0; i < cc - (cc % 3); i++){
			oc = constraints[i + cr*numConstraints];

			if (c.isRay && oc.isRay){
				point = intersectRayRay(c.ray, oc.ray);
			}
			else if (c.isRay && !oc.isRay){
				point = intersectRaySegment(c.ray, oc.ray.pos, oc.endpoint);
			}
			else if (!c.isRay && oc.isRay){
				point = intersectRaySegment(oc.ray, c.ray.pos, c.endpoint);
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
			point.distToOrigin = glm::distance(point.point, c.ray.pos);
			intersections[cc*(numConstraints - 1) + j + cr*numIntersections] = point;
			j++;
		}

		for (int i = cc + (3 - cc % 3); i < numConstraints; i++){
			oc = constraints[i + cr*numConstraints];

			if (c.isRay && oc.isRay){
				point = intersectRayRay(c.ray, oc.ray);
			}
			else if (c.isRay && !oc.isRay){
				point = intersectRaySegment(c.ray, oc.ray.pos, oc.endpoint);
			}
			else if (!c.isRay && oc.isRay){
				point = intersectRaySegment(oc.ray, c.ray.pos, c.endpoint);
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
			point.distToOrigin = glm::distance(point.point, c.ray.pos);
			intersections[cc*(numConstraints - 1) + j + cr*numIntersections] = point;
			j++;
		}
	}
}

__global__ void kernLabelInsideOutIntersections(int totIntersections, int numIntersections, int numAgents, int numFVOs, intersection* intersections, FVO* fvos){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totIntersections){
		int ir = index / numIntersections;
		int ic = index % numIntersections;

		if (!intersections[index].isIntersection){
			return;
		}

		bool isOutside = true;

		//TODO: can we prevent ourselves from checking against our own FVO?
		for (int i = 0; i < numFVOs; i++){
			if (pointInFVO(fvos[ir*numFVOs + i], intersections[index].point)){
				isOutside = false;
				break;
			}
		}
		intersections[index].isOutside = isOutside;
	}
}

__global__ void kernSortIntersectionPoints(int totConstraints, int numAgents, int numConstraints, int numIntersections, intersection* intersections, constraint* constraints){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totConstraints){
		int intersectionPerConstraint = numConstraints - 1;
		int cr = index / numConstraints;
		int cc = index % numConstraints;

		thrust::sort(thrust::seq, &intersections[cr*numIntersections + cc*intersectionPerConstraint], &intersections[cr*numIntersections + cc*intersectionPerConstraint] + intersectionPerConstraint, DistToOriginComp());
	}
}

__global__ void kernLabelInsideOutSegments(int totConstraints, int numAgents, int numConstraints, int numIntersections, glm::vec3* vel_new, int* min_vel_diff, intersection* intersections, constraint* constraints, agent* agents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < totConstraints){
		int intersectionPerConstraint = numConstraints - 1;
		int cr = index / numConstraints;
		int cc = index % numConstraints;

		//if (cc % 3 == 1){
		//	return;
		//}

		glm::vec3 vel_pos_des = agents[cr].pos + agents[cr].vel;

		intersection in;
		intersection endpoint1 = intersections[cr*numIntersections + cc*intersectionPerConstraint];
		intersection endpoint2;
		bool isOutside;
		bool first = true;

		float old;
		float dist_new;
		glm::vec3 vel_pos_new;

		for (int i = 1; i < intersectionPerConstraint; i++){
			in = intersections[cr*numIntersections + cc*intersectionPerConstraint + i];
			if (!in.isIntersection) continue;
			endpoint2 = in;

			if (first && endpoint1.isOutside && endpoint2.isOutside){
				isOutside = true;
				first = false;
			}
			else if (endpoint1.isOutside && endpoint2.isOutside && !isOutside){
				isOutside = true;
			}
			else{
				isOutside = false;
			}

			if (isOutside){
				vel_pos_new = projectPointToSegment(endpoint1.point, endpoint2.point, vel_pos_des);
				dist_new = glm::distance(vel_pos_des, vel_pos_new);
				old = atomicMin(&min_vel_diff[cr], __float_as_int(dist_new));

				if (min_vel_diff[cr] == __float_as_int(dist_new)){
					vel_new[cr] = vel_pos_new - agents[cr].pos;
				}
			}

			endpoint1 = endpoint2;
		}

		if ((first && endpoint1.isOutside) || (endpoint1.isOutside && !isOutside)){
			ray cray;
			cray.pos = endpoint1.point;
			cray.dir = constraints[index].ray.dir;

			vel_pos_new = projectPointToRay(cray, vel_pos_des);
			dist_new = glm::distance(vel_pos_des, vel_pos_new);
			old = atomicMin(&min_vel_diff[cr], __float_as_int(dist_new));
			if (min_vel_diff[cr] == __float_as_int(dist_new)){
				vel_new[cr] = vel_pos_new - agents[cr].pos;
			}
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
	numFVOs = numNeighbors;
	numConstraints = numFVOs * 3; // number of constraints that each robot has to compare
	numIntersections = numConstraints * (numConstraints - 1); //-1 to remove self-constraint
	
	totNeighbors = numNeighbors * numAgents;
	totFVOs = numFVOs * numAgents;
	totConstraints = numConstraints * numAgents;
	totIntersections = numIntersections * numAgents;

	totSamples = numAgents * NUM_SAMPLES;

	numHRVOs = numNeighbors;
	totHRVOs = numAgents * numHRVOs;
	int numNaiveCandidates = 2 * numHRVOs;
	int numIntersectionCandidates = 4 * numHRVOs*(numHRVOs - 1);
	numCandidates = numNaiveCandidates + numIntersectionCandidates;
	totCandidates = numAgents * numCandidates;

	// Get the number of blocks we need
	dim3 fullBlocksForFVOs((totFVOs + blockSize - 1) / blockSize);
	dim3 fullBlocksForConstraints((totConstraints + blockSize - 1) / blockSize);
	dim3 fullBlocksForUG((GRIDMAX*GRIDMAX + blockSize - 1) / blockSize);
	dim3 fullBlocksForIntersections((totIntersections + blockSize - 1) / blockSize);
	dim3 fullBlocksForSamples((totSamples + blockSize - 1) / blockSize);
	dim3 fullBlocksForHRVOs((totHRVOs + blockSize - 1) / blockSize);
	dim3 fullBlocksForCandidates((totCandidates + blockSize - 1) / blockSize);

	// Free everything
	cudaFree(dev_neighbors);
	cudaFree(dev_fvos);
	cudaFree(dev_endpoints);
	
	cudaFree(dev_intersections);
	cudaFree(dev_constraints);

	cudaFree(dev_min_vel_diff);
	cudaFree(dev_vel_new);

	cudaFree(dev_sample_vels);
	cudaFree(dev_scores);

	cudaFree(dev_hrvos);
	cudaFree(dev_candidates);
	
	// UG
	cudaFree(dev_uglist);
	cudaFree(dev_startIdx);
	cudaFree(dev_ug_neighbors);
	cudaFree(dev_num_neighbors);
	
	cudaMalloc((void**)&dev_neighbors, totFVOs*sizeof(int));
	cudaMalloc((void**)&dev_fvos, totFVOs*sizeof(FVO));
	cudaMalloc((void**)&dev_endpoints, 6*totFVOs*sizeof(glm::vec2));

	cudaMalloc((void**)&dev_constraints, totConstraints*sizeof(constraint));
	cudaMalloc((void**)&dev_intersections, totIntersections*sizeof(intersection));

	cudaMalloc((void**)&dev_min_vel_diff, numAgents*sizeof(int));
	cudaMalloc((void**)&dev_vel_new, numAgents*sizeof(glm::vec3));
	
	// UG
	cudaMalloc((void**)&dev_uglist, numAgents*sizeof(UGEntry));
	cudaMalloc((void**)&dev_startIdx, GRIDMAX*GRIDMAX*sizeof(int));
	cudaMalloc((void**)&dev_ug_neighbors, numNeighbors*numAgents*sizeof(int));
	cudaMalloc((void**)&dev_num_neighbors, numAgents*sizeof(int));

	cudaMalloc((void**)&dev_sample_vels, totSamples*sizeof(glm::vec3));
	cudaMalloc((void**)&dev_scores, totSamples*sizeof(float));

	cudaMalloc((void**)&dev_hrvos, totHRVOs*sizeof(HRVO));
	cudaMalloc((void**)&dev_candidates, totCandidates*sizeof(CandidateVel));
	
	// TODO: this should actually be a kernel initialization
	cudaMemset(dev_in_pcr, false, numAgents*sizeof(bool));
	cudaThreadSynchronize();

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
	
#ifdef DEBUG_UG
	UGEntry* hst_uglist = (UGEntry*)malloc(numAgents * sizeof(UGEntry));
	cudaMemcpy(hst_uglist, dev_uglist, numAgents*sizeof(UGEntry), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numAgents; i++){
		printf("(cell %d, agent %d)\n", hst_uglist[i].cellId, hst_uglist[i].agentId);
	}

	printf("---\n");
	int* hst_startIdx = (int*)malloc(GRIDMAX*GRIDMAX*sizeof(int));
	cudaMemcpy(hst_startIdx, dev_startIdx, GRIDMAX*GRIDMAX*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < GRIDMAX*GRIDMAX; i++){
		if (hst_startIdx[i] >= 0){
			printf("cell: %d, start: %d\n", i, hst_startIdx[i]);
		}
	}

	free(hst_uglist);
	free(hst_startIdx);

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
#endif

#ifdef USE_HRVO
	kernComputeHRVOs<<<fullBlocksPerGrid, blockSize>>>(totHRVOs, numHRVOs, numAgents, dev_hrvos, dev_agents, dev_neighbors, dt);

	kernInitCandidateVels<<<fullBlocksForCandidates, blockSize>>>(totCandidates, dev_candidates);

	kernComputeNaiveCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, numAgents, numHRVOs, numCandidates, dev_candidates, dev_hrvos, dev_agents);

	kernComputeIntersectionCandidateVels<<<fullBlocksForHRVOs, blockSize>>>(totHRVOs, numAgents, numHRVOs, numCandidates, numNaiveCandidates, dev_candidates, dev_hrvos, dev_agents);

	//CandidateVel* hst_candidates = (CandidateVel*)malloc(totCandidates);

	kernComputeValidCandidateVels<<<fullBlocksForCandidates, blockSize>>>(totCandidates, numAgents, numHRVOs, numCandidates, dev_candidates, dev_hrvos, dev_agents);

	kernComputeBestVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, numCandidates, dev_vel_new, dev_candidates);

	kernComputeInPCR<<<fullBlocksPerGrid, blockSize>>>(numAgents, numHRVOs, dev_in_pcr, dev_hrvos, dev_agents);

	kernUpdateVel<<<fullBlocksPerGrid,blockSize>>>(numAgents, dev_agents, dev_in_pcr, dev_vel_new);
#endif

#ifdef USE_FVO
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
	free(hst_intersection);

	// Compute Inside/Outside Points
	// TODO: Need to do compaction on these
	//kernLabelInsideOutIntersections<<<fullBlocksForIntersections, blockSize>>>(totIntersections, numIntersections, numAgents, numFVOs, dev_intersections, dev_fvos);

	// Sort Intersection Points based on distance from endpoint
	//kernSortIntersectionPoints<<<fullBlocksForConstraints, blockSize>>>(totConstraints, numAgents, numConstraints, numIntersections, dev_intersections, dev_constraints);

	// Compute Inside/Outside Line Segments, track the nearest velocities
	//kernInitMinVelDiff<<<fullBlocksPerGrid, blockSize>>>(numAgents, dev_min_vel_diff);
	//kernLabelInsideOutSegments<<<fullBlocksForConstraints, blockSize>>>(totConstraints, numAgents, numConstraints, numIntersections, dev_vel_new, dev_min_vel_diff, dev_intersections, dev_constraints, dev_agents);

	/*
	int* hst_min_vel_diff = (int*)malloc(numAgents*sizeof(int));
	printf("Minimum velocity difference: ");
	cudaMemcpy(hst_min_vel_diff, dev_min_vel_diff, numAgents*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numAgents; i++){
		printf("%d ", hst_min_vel_diff[i]);
	}
	printf("\n");
	*/

	// Update the velocities according to ClearPath
	//kernUpdateVel2Only<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_fvos, dev_in_pcr);
	//kernUpdateVelBad << <fullBlocksPerGrid, blockSize >> >(numAgents, dt, dev_agents, dev_fvos, dev_in_pcr);
	//kernUpdateVel<<<fullBlocksPerGrid,blockSize>>>(numAgents, dev_agents, dev_in_pcr, dev_vel_new);
#endif

#ifdef USE_SAMPLING
	kernSampleVelocities<<<fullBlocksForSamples, blockSize>>>(totSamples, iter+1, dev_sample_vels);

	kernScoreSamples<<<fullBlocksForSamples, blockSize>>>(totSamples, numAgents, NUM_SAMPLES, dev_scores, dev_sample_vels, dev_agents);

	kernSelectVel<<<fullBlocksPerGrid, blockSize>>>(numAgents, NUM_SAMPLES, dev_scores, dev_sample_vels, dev_agents, dev_in_pcr);
#endif

	agent* hst_agents = (agent*)malloc(numAgents*sizeof(agent));
	cudaMemcpy(hst_agents, dev_agents, numAgents*sizeof(agent), cudaMemcpyDeviceToHost);
	printf("Magnitude of velocities? ");
	for(int i = 0; i < numAgents; i++){
		printf("%f ", glm::length(hst_agents[i].vel));
	}
	printf("\n");

	// Update the positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numAgents, dt, dev_agents, dev_pos);


}
