cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD  11)

set(CMAKE_CUDA_COMPILER //usr/local/cuda-8.0/bin/nvcc)
set(CMAKE_CUDA_COMPILER_ENV_VAR //usr/local/cuda-8.0/bin/ )
project(norigl)

find_package(CUDA QUIET REQUIRED)


add_library(pugixml STATIC ../ext/pugixml/src/pugixml.cpp include/nori/arealight.h)
set_target_properties(pugixml PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ext_build/dist/lib")


find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

include_directories(../ext/pugixml/src)



find_package(Eigen3 REQUIRED) #(requires 3.1.0 or greater)
include( ${EIGEN3_USE_FILE} )


find_package (Threads)
find_package( OpenCV REQUIRED )
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        #--relocatable-device-code=true
	-O3 -use_fast_math -keep -lineinfo --keep-dir /tmp/ --maxrregcount 64 --ptxas-options  -warn-double-usage,-v -gencode arch=compute_61,code=sm_61 -std=c++11
)
#set( CUDA_SEPARABLE_COMPILATION ON)


set(CMAKE_EXTERNAL_ARGUMENTS -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_GENERATOR=${CMAKE_GENERATOR} -Wno-dev
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/ext_build/dist)
include(ExternalProject)
# Build Intel Thread Building Blocks
ExternalProject_Add(tbb_p
  URL             "${CMAKE_SOURCE_DIR}/tbb"
  PREFIX          "ext_build"
  CMAKE_ARGS      ${CMAKE_EXTERNAL_ARGUMENTS}
)
# Compile against & link to previously compiled external projects
link_directories(${CMAKE_BINARY_DIR}/ext_build/dist/lib)
include_directories(
  ${CMAKE_BINARY_DIR}/ext_build/dist/include
)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES
    main.cpp
       MainWindow.h  ShaderProgramm.h  CudaRender.h
        include/filesystem/resolver.h
        include/filesystem/path.h
        include/pcg32.h
        include/nori/scene.h
        include/tinyformat.h
        include/nori/bbox.h
        include/nori/bsdf.h
        include/nori/bvh.h
        include/nori/camera.h
        include/nori/color.h
        include/nori/common.h
        include/nori/frame.h
        include/nori/dpdf.h
        include/nori/integrator.h
        include/nori/emitter.h
        include/nori/mesh.h
        include/nori/object.h
        include/nori/parser.h
        include/nori/perspective.h
        include/nori/proplist.h
        include/nori/transform.h
        include/nori/texture.h
        include/nori/shape.h
        include/nori/vector.h
        include/nori/ray.h
        include/nori/warp.h
		include/nori/parser.h
        include/nori/independent.h
        include/nori/timer.h
        include/nori/diffuse.h
        include/nori/dielectric.h
        include/nori/consttexture.h
		include/nori/imagetexture.h
		include/nori/sphere.h
        include/nori/normalModifier.h
        include/nori/bumpMap.h
        include/nori/normalMap.h
        finalKernel.cu

        #nori/arealight.cpp.cu
        #nori/perspective.cpp.cu
        #nori/bvh.cpp.cu
        #nori/diffuse.cpp.cu
        #nori/independent.cpp.cu
        #nori/pathMisIntegrator.cpp.cu
        #nori/warp.cpp.cu
        #nori/common.cpp.cu
        #nori/mesh.cpp.cu
        #nori/dielectric.cpp.cu
		#renderer.cc.cu


 		nori/obj.cpp
        nori/consttexture.cpp
		nori/shape.cpp
		nori/scene.cpp
        nori/parser.cpp
        nori/proplist.cpp

        nori/object.cpp
        #nori/diffuse.cpp
        #nori/dielectric.cpp

        MainWindow.cpp
        glad.c
        ShaderProgramm.cpp
        CudaRender.cpp)
add_definitions(-DEIGEN_DONT_ALIGN)
link_directories(lib)
include_directories(include)
include_directories(tbb/include)

cuda_add_executable(norigl ${SOURCE_FILES})
add_dependencies(norigl tbb_p)
add_dependencies(norigl pugixml)

target_link_libraries(norigl ${GLFW_LIBRARIES}  ${OpenCV_LIBS} dl pugixml tbb  ${CMAKE_THREAD_LIBS_INIT})
