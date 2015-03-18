cmake_minimum_required(VERSION 2.8.4)

project(DSHIN_SHAPE_CPP)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(MATLAB REQUIRED)
find_package(OpenMP REQUIRED)

if(MATLAB_FOUND)
  MESSAGE(STATUS "Matlab found at ${MATLAB_ROOT}")
else(MATLAB_FOUND)
  MESSAGE(FATAL_ERROR "Matlab not found")
endif(MATLAB_FOUND)

set(CMAKE_VERBOSE_MAKEFILE on)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(MEX_OUTPUT_DIR "${PROJECT_SOURCE_DIR}")
set(MESH_CONN_COMPS_MEX "meshConnComps${MATLAB_MEX_EXTENSION}")

add_custom_target(
    "${PROJECT_NAME}_MEX" ALL
    DEPENDS ${MEX_OUTPUT_DIR}/${MESH_CONN_COMPS_MEX}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(MEX_BUILD_FLAGS -g -v)
else()
    set(MEX_BUILD_FLAGS -O -v -DNDEBUG -DRELEASE)
endif()

add_custom_command(
    OUTPUT ${MEX_OUTPUT_DIR}/${MESH_CONN_COMPS_MEX}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND ${MATLAB_MEX_EXECUTABLE}
    meshConnComps.mex.cc
    LDFLAGS=\"\\$$LDFLAGS ${OpenMP_CXX_FLAGS} -std=c++0x -Ofast\"
    CXXFLAGS=\"\\$$CXXFLAGS ${OpenMP_CXX_FLAGS} -std=c++0x -Ofast\"
    -largeArrayDims -cxx
    -outdir ${MEX_OUTPUT_DIR} -output ${MESH_CONN_COMPS_MEX}
    -L${LIBRARY_OUTPUT_PATH} ${MEX_BUILD_FLAGS}
)
