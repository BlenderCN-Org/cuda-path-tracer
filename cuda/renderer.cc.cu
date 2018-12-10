#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <nori/scene.h>
#include <nori/camera.h>
#include <device_functions.h>
#include <nori/independent.h>
#include <nori/perspective.h>
#include <nori/integrator.h>
#include <nori/pathMisIntegrator.h>
#include <nori/bvh.h>
#include <chrono>
#include <unistd.h>

class Perspective;
NORI_NAMESPACE_BEGIN

//__shared__ Sampler samplerS[1024];
#define MEDIAN(a,b,c) ((a-b)*(b-c) > -1 ? b : ((a-b)*(a-c) < 1 ? a : c))
__global__ void render_gpu(cudaSurfaceObject_t cuda_data, int width, int height, nori::Scene* scene,Color3f* image,size_t step)
{

    Sampler s;// = samplerS[threadIdx.x * blockDim.y + threadIdx.y]
    int iterations = scene->getSampler()->m_sampleCount;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;  x += blockDim.x * gridDim.x) {
       for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;  y += blockDim.y * gridDim.y) {
           // if (y <= 100) break;

           //if (x != 5 || y != 500) asm("exit;");
            int gIn = x*height+y;

            Independent::prepare(s, Point2i(x+width*step, y+height*step));
            nori::Point2f p((float(x)), (float(y)));
            Ray3f r;
            const nori::Camera *c = scene->getCamera();
            CallCamera(c, sampleRay, r, p, CallSampler(s, next2D));
            /**todo we could use constant memory for the origian of the ray**/
            Point3f rO = r.o;
            Vector3f rD = r.d;
            Integrator *i = scene->getIntegrator();
            //costly but needed, check whether the camera ray hits something, if not this thread won't be useful
            //if(!scene->rayIntersect(r))
            //   asm("exit;");




           Color3f color(0);;
           int in = 0;
           Ray3f nr = r;
           Color3f t = 1;
           float lastBsdfPdf = 1;
           bool  lastBsdfDelta = true;


           while(in<iterations){
                color += CallIntegrator(i, Li, scene, s, nr,t,lastBsdfDelta,lastBsdfPdf);
                //color += float(x+y)/(800.0f*600);
                //in++;;
                if(t(0)<0){
                    nr.o = rO;
                    nr.d = rD;
                    nr.update();
                    t = 1;
                    lastBsdfDelta = true;
                    
                    in++;
                }
            }

            color /= iterations;
            /*if (step) {
                color = (color - image[gIn]) / (step + 1) + image[gIn];
            }*/

           // Color3f colorO = color.toSRGB();
            surf2Dwrite(make_float4(color(0), color(1), color(2), 1.0f),
                        cuda_data, (int) sizeof(float4) * x, height - y,
                        cudaBoundaryModeClamp);

            //image[gIn].x() = color.x();
            image[gIn] = color.toSRGB();
        }
    }
}

/**
 *
 * @param resource
 * @param w
 * @param h
 * @param scene  scene object on GPU memory
 * @param sampler  Sampler object on CPU memory
 */
static Color3f* image = nullptr;
//static nori::Independent* gpuSampler = nullptr;
static size_t step = 0;
static std::chrono::milliseconds startTime;

void render_scene(cudaSurfaceObject_t resource, int w, int h,nori::Scene *scene, std::string filename)
{
    int blockSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&gridSize,&blockSize,render_gpu,0,w*h);

    //we want to render 2D blocks not lines
    int blockW = sqrt(blockSize);
    int blockH = blockSize/blockW;
    dim3 block(blockW,blockH);
    int gridSizeW = (w + blockW - 1) / blockW;
    int gridSizeH = (h + blockH - 1) / blockH;
    dim3 grid(gridSizeW,gridSizeH);
    //At the moment we do not use the given Sampler but simply independent
    if(!image) {
        startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        cudaMalloc((void **) &image, w * h * sizeof(Color3f));
        //cudaMemset(image, 0, w * h * sizeof(Color3f));
    }

    //number of kernel runs
    const int sCount = 1;
    //int block = 64;
    //int grid  = (w*h + block - 1) / block;
    if(step<sCount) {
        nori::render_gpu << < grid,block >> > (resource, w, h, scene,  image, step);
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
        std::cout<<step<<std::endl;
        step++;
    }else if(sCount == step){
       if(cudaSuccess==cudaStreamQuery(0)){
            cudaDeviceSynchronize();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - startTime;
            std::cout << "finished: Took" << float(diff.count())/1000  << " seconds!"<< std::endl ;
            step++;


             //load image from GPU


            size_t lastindex = filename.find_last_of("."); 
            std::string name = filename.substr(0, lastindex);

            cv::Mat convertedImage(cv::Size(h,w),CV_8UC3);
            cv::Mat mImage(cv::Size(h,w),CV_32FC3);
            cudaMemcpy(mImage.data,image,sizeof(Color3f)*w*h,cudaMemcpyDeviceToHost);
            cv::cvtColor(mImage,mImage,CV_RGB2BGR);
            mImage.convertTo(convertedImage,CV_8UC3,255);
            cv::transpose(convertedImage,convertedImage);
            cv::imwrite(name + ".png",convertedImage);

            filter_scene(resource,w,h,scene,image);

            cudaMemcpy(mImage.data,image,sizeof(Color3f)*w*h,cudaMemcpyDeviceToHost);
            cv::cvtColor(mImage,mImage,CV_RGB2BGR);
            mImage.convertTo(convertedImage,CV_8UC3,255);
            cv::transpose(convertedImage,convertedImage);
            cv::imwrite(name + "_filtered.png",convertedImage);


           step++;
           //exit(0);
       }else{
           sleep(1);
       }
    }

}

NORI_NAMESPACE_END
