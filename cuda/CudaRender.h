
#ifndef OPENGL_CUDARENDER_H
#define OPENGL_CUDARENDER_H
#include <GL/gl.h>
#include <nori/common.h>
#include <cuda_runtime_api.h>
#include <chrono>
class CudaRender {
public:
	CudaRender();
    ~CudaRender();

/*One integration step performed here
 *
 * */
void step();

    nori::Vector2i loadFile(char *string);

	void registerTexture(GLuint t);
private:
	nori::Scene* gpuScene;
	struct cudaGraphicsResource* tex_CUDA;


	std::chrono::milliseconds startTime;
    cudaStream_t cuda_stream;
    cudaSurfaceObject_t textureSurface;
    int w;
	int h;

    std::string filename;

};
#endif
