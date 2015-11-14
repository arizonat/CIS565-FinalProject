set(SOURCE_FILES
    "cudaMat4.hpp"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "utilityCore.hpp"
    "utilityCore.cpp"
    "kernel.h"
    "kernel.cu"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
