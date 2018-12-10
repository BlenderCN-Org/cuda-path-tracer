#include <cuda_gl_interop.h>
#include <cstdio>
#include <iostream>
#include <thread>
#include <nori/common.h>
#include <filesystem/path.h>
#include <filesystem/resolver.h>
#include "CudaRender.h"
#include "renderer.h"
#include "nori/parser.h"
#include <nori/scene.h>
#include <nori/camera.h>


CudaRender::CudaRender(){

    cudaSetDevice(0);
    cudaGLSetGLDevice(0);
    //Create CUDA stream
    CHECK_ERROR(cudaStreamCreate(&cuda_stream));
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
}

void CudaRender::step(){
    nori::render_scene(textureSurface,w,h,gpuScene, filename);

}

CudaRender::~CudaRender(){

    //this also frees the device memory
    CHECK_ERROR(cudaGraphicsUnmapResources(1, &tex_CUDA, cuda_stream));
    //Destroy the CUDA stream
    CHECK_ERROR(cudaStreamDestroy(cuda_stream));

}

void CudaRender::registerTexture(GLuint t){


    CHECK_ERROR(cudaGraphicsGLRegisterImage (&tex_CUDA,t,GL_TEXTURE_2D,cudaGraphicsRegisterFlagsWriteDiscard));

    //Map the graphics resource to the CUDA stream
    CHECK_ERROR(cudaGraphicsMapResources(1, &tex_CUDA, cuda_stream));


    //okay now we want to get the first texture out of our texture resource with minmap id 0
    cudaArray_t textureArray;
    CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&textureArray, tex_CUDA, 0, 0));


    cudaResourceDesc textureDesc;
    textureDesc.resType = cudaResourceTypeArray;
    textureDesc.res.array.array = textureArray;

    //and create a surface object such that the kernels can read + write to it

    CHECK_ERROR(cudaCreateSurfaceObject(&textureSurface, &textureDesc));


}
nori::Vector2i CudaRender::loadFile(char *string) {
    filename = std::string(string);
    filesystem::path f(filename);

    //load filename using the original Nori XML parser
    nori::getFileResolver()->prepend(f.parent_path());
    auto ret = nori::loadFromXML(filename);


    //Extract image size information from parsed xml
    auto s = (nori::Scene*)ret;
    auto size = s->getCamera()->getOutputSize();

    //save objectID from scene Object
    auto sceneID = ret->objectId;


    //allocate memory for objects on gpu
    nori::NoriObject* h;
    h = nori::NoriObject::head;
    nori::NoriObject* gpuObject[nori::NoriObject::count];
    while(h){

            void *p  = NULL;
            cudaMalloc(&p,h->getSize());
            gpuObject[h->objectId] = static_cast<nori::NoriObject *>(p);
            h = h->next;
    }


    //actually move objects to gpu
    h = nori::NoriObject::head;
    while(h){
            h->gpuTransfer(gpuObject);
            cudaMemcpy(gpuObject[h->objectId],h,h->getSize(),cudaMemcpyHostToDevice);
            h = h->next;
    }


    //remove objects from cpu
    h = nori::NoriObject::head;
    while(h){
        auto tmp = h;
        h = h->next;
        operator delete(tmp);
    }

    //store scene pointer (to GPU memory) and size information
    gpuScene = (nori::Scene *) gpuObject[sceneID];
    this->w = size.x();
    this->h = size.y();
    return size;



    /**
     * Todo safely remove the objects from cpu memory (can't call constructor as we assume that they do not hold any valid pointer)
     */


}

