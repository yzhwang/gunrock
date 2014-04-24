project(gunrock)

set(gunrock_VERSION_MAJOR 0)
set(gunrock_VERSION_MINOR 1)
set(gunrock_VERSION_PATCH 0)

set(gunrock_REQUIRED_BOOST_VERSION 1.53)

cmake_minimum_required(VERSION 2.8)

option(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA REQUIRED)

if(CUDA_64_BIT_DEVICE_CODE)
  set(gunrock_arch_suffix x86_64)
else()
  set(gunrock_arch_suffix i386)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

find_package(BoostHeaders ${gunrock_REQUIRED_BOOST_VERSION} REQUIRED)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif (Boost_FOUND)


set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)

#set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX

option(CUDPP_GENCODE_SM10
       "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
       OFF)

option(CUDPP_GENCODE_SM13
       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
       OFF)

option(CUDPP_GENCODE_SM20
       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
       OFF)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       ON)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       OFF)

option(CUDPP_GENCODE_SM50
       "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750 TI)"
       OFF)

option(CUDA_VERBOSE_PTXAS
       "On to enable verbose output from the PTXAS assembler."
       OFF
  )

if (CUDPP_GENCODE_SM10)
  set(GENCODE ${GENCODE} ${GENCODE_SM10})
endif(CUDPP_GENCODE_SM10)

if (CUDPP_GENCODE_SM13)
  set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(CUDPP_GENCODE_SM13)

if (CUDPP_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(CUDPP_GENCODE_SM20)

if (CUDPP_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(CUDPP_GENCODE_SM30)

if (CUDPP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(CUDPP_GENCODE_SM35)

if (CUDPP_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(CUDPP_GENCODE_SM50)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

if (WIN32)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler /bigobj -Xcompiler /Zm500)
endif (WIN32)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CUDA_HOST_COMPILER /usr/bin/clang)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g;-G)

set (CCFILES simple_example.cu cpu_graph_lib.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

CUDA_ADD_EXECUTABLE(simple_example_${CUDA_VERSION_STRING}_${gunrock_arch_suffix}
  ${CCFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
