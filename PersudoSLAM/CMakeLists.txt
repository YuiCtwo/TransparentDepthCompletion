cmake_minimum_required(VERSION 3.22)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
project(PersudoSLAM)
find_package(PythonInterp REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
set(CMAKE_CXX_STANDARD 17)

set(Pangolin_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Pangolin/build/src" CACHE PATH "Pangolin build directory")
#set(Pangolin_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Pangolin/include" CACHE PATH "Pangolin Include directory")
#set(Pangolin_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Pangolin/Bin/x64-Release/libOpenNI2.so" CACHE PATH "OpenNI2 library")
set(OPENNI2_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/OpenNI2/Include" CACHE PATH "OpenNI2 Include directory")
set(OPENNI2_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third-party/OpenNI2/Bin/x64-Release/libOpenNI2.so" CACHE PATH "OpenNI2 library")
#set(Sophus_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Sophus" CACHE PATH "Sophus Include directory")
set(efusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Core/Shaders" CACHE PATH "Where the shaders live")

# set(CUDNN_INCLUDE_DIR "/usr/local/cuda/include")
# set(CUDNN_LIBRARY "/usr/local/cuda/lib64")
#set(CMAKE_PREFIX_PATH "/home/ctwo/repo/libtorch/share/cmake/Torch")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-DCMAKE_PREFIX_PATH=/home/ctwo/repo/libtorch/")
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(ZLIB REQUIRED)
find_package(JPEG REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(SuiteSparse REQUIRED)

find_package(OpenCV 3.4 REQUIRED)
find_package(Torch REQUIRED
             PATHS /home/ctwo/repo/libtorch)

message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIR})
#include_directories(${OPENCV_INCLUDE_DIRS})

if(WITH_REALSENSE)
  include_directories(${REALSENSE_INCLUDE_DIR})
  add_definitions(-DWITH_REALSENSE)
  set(EXTRA_LIBS ${EXTRA_LIBS} ${REALSENSE_LIBRARY})
endif()

file(GLOB srcs *.cpp *.h *.cu *.cuh)
file(GLOB tools_srcs Tools/*.cpp Tools/*.h Tools/*.cu Tools/*.cuh)

add_definitions(-Dlinux=1)
add_definitions(-DEIGEN_MAX_ALIGN_BYTES=0)
add_definitions(-DEIGEN_MAX_STATIC_ALIGN_BYTES=0)


set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3 -Wall -DSHADER_DIR=${efusion_SHADER_DIR}  -DFMT_HEADER_ONLY -lfmt")

add_subdirectory(Core)
add_executable(PersudoSLAM
               ${tools_srcs}
               ${srcs}
)

target_link_libraries(PersudoSLAM
                      efusion
                      ${ZLIB_LIBRARY}
                      ${JPEG_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${EXTRA_LIBS}
                      ${OPENNI2_LIBRARY}
                      ${SUITESPARSE_LIBRARIES}
                      ${BLAS_LIBRARIES}
                      ${LAPACK_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      ${TORCH_LIBRARIES}
)