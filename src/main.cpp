/**
 * @file      main.cpp
 * @brief     Example N-body simulation for CIS 565
 * @authors   Liam Boone, Kai Ninomiya
 * @date      2013-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"

// ================
// Configuration
// ================

#define VISUALIZE 1

const int N_FOR_VIS = 2;
const float DT = 0.2f;

glm::vec3* hst_pos;

/**
 * C main function.
 */
int main(int argc, char* argv[]) {
    projectName = "565 CUDA Intro: N-Body";

    if (init(argc, argv)) {
        mainLoop();
        return 0;
    } else {
        return 1;
    }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
 * Initialization of CUDA and GLFW.
 */
bool init(int argc, char **argv) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
                << "Error: GPU device number is greater than the number of devices!"
                << " Perhaps a CUDA-capable GPU is not installed?"
                << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();

    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
        return false;
    }
    int width = 1280;
    int height = 720;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    
	// TODO: original lines
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_FALSE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize drawing state
    initVAO();

    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    cudaGLRegisterBufferObject(planetVBO);
    // Initialize N-body simulation
    Nbody::initSimulation(N_FOR_VIS);

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    //glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(0), glm::vec3(0, 0, 1));
	glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(0), glm::vec3(0, 1, 0));

    projection = projection * view;

    initShaders(program);

    glEnable(GL_DEPTH_TEST);

	hst_pos = (glm::vec3*)malloc(N_FOR_VIS*sizeof(glm::vec3));

    return true;
}

void initVAO() {
    glm::vec4 vertices[] = {
        glm::vec4( -1.0, -1.0, 0.0, 0.0 ),
        glm::vec4( -1.0,  1.0, 0.0, 0.0 ),
        glm::vec4(  1.0,  1.0, 0.0, 0.0 ),
        glm::vec4(  1.0, -1.0, 0.0, 0.0 ),
    };

    GLuint indices[] = { 0, 1, 2, 1, 2, 3 };

    GLfloat *bodies    = new GLfloat[4 * (N_FOR_VIS + 1)];
    GLuint *bindices   = new GLuint[N_FOR_VIS + 1];

    glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
    glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

    for (int i = 0; i < N_FOR_VIS + 1; i++) {
        bodies[4 * i + 0] = 0.0f;
        bodies[4 * i + 1] = 0.0f;
        bodies[4 * i + 2] = 0.0f;
        bodies[4 * i + 3] = 1.0f;
        bindices[i] = i;
    }

    glGenVertexArrays(1, &planetVAO);
    glGenBuffers(1, &planetVBO);
    glGenBuffers(1, &planetIBO);

    glBindVertexArray(planetVAO);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS + 1) * sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS + 1) * sizeof(GLuint), bindices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(positionLocation);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);

    delete[] bodies;
    delete[] bindices;
}

void initShaders(GLuint * program) {
    GLint location;

    program[PROG_PLANET] = glslUtility::createProgram(
                     "shaders/planet.vert.glsl",
                     "shaders/planet.geom.glsl",
                     "shaders/planet.frag.glsl", attributeLocations, 1);
    glUseProgram(program[PROG_PLANET]);

    if ((location = glGetUniformLocation(program[PROG_PLANET], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_PLANET], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }

	program[PROG_LINE] = glslUtility::createProgram("shaders/line.vert.glsl", 
													 NULL, 
													 "shaders/line.frag.glsl", attributeLocations, 1);
	//glUseProgram(program[PROG_LINE]);

	hst_endpoints = (glm::vec2*)malloc(6 * (N_FOR_VIS)*sizeof(glm::vec2));
}

//====================================
// Main loop
//====================================
void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4 *dptr = NULL;
    float *dptrvert = NULL;
    cudaGLMapBufferObject((void**)&dptrvert, planetVBO);

    // execute the kernel
    Nbody::stepSimulation(DT);
#if VISUALIZE
    Nbody::copyPlanetsToVBO(dptrvert, hst_endpoints, hst_pos);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
}

void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();

        if (time - timebase > 1.0) {
            fps = frame / (time - timebase);
            timebase = time;
            frame = 0;
        }

        runCUDA();

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#if VISUALIZE
        glUseProgram(program[PROG_PLANET]);
        glBindVertexArray(planetVAO);
        glPointSize(2.0f);
        glDrawElements(GL_POINTS, N_FOR_VIS, GL_UNSIGNED_INT, 0);
        glPointSize(1.0f);

		//glUseProgram(program[PROG_LINE]);
		glUseProgram(0);

		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(&projection[0][0]);

		glBegin(GL_LINES);

		glVertex2f(hst_pos[0].x,hst_pos[0].y);
		glVertex2f(hst_pos[0].x+2, hst_pos[0].y+2);

		for (int i = 0; i < 6*(N_FOR_VIS); i++){
			glVertex2f(hst_endpoints[i].x, hst_endpoints[i].y);
			printf("%f, %f\n",hst_endpoints[i].x,hst_endpoints[i].y);
		}
		glEnd();
        glUseProgram(0);
        glBindVertexArray(0);
#endif

        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}


void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

