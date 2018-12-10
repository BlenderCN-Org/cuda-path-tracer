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

NORI_NAMESPACE_BEGIN


#define IDX2(i,j,N) (((i)*(N))+(j))
#define COL_NORMSQ(c) (c.x()*c.x() + c.y()*c.y() + c.z()*c.z())

__global__ void copy_surface_to_data(cudaSurfaceObject_t cuda_data, int width, int height, Color3f* data)
{
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;  x += blockDim.x * gridDim.x) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;  y += blockDim.y * gridDim.y) {

            
            float4 val;            
            surf2Dread(&val, cuda_data,(int) sizeof(float4) * x, height - y,cudaBoundaryModeClamp);
            data[IDX2(x,y,height)] = Color3f(val.x, val.y, val.z);
        }
    }
}

__global__ void copy_weighted_data_to_surface(cudaSurfaceObject_t cuda_data, Color3f* image, int width, int height, Color3f* flt, float* wgtsum)
{
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;  x += blockDim.x * gridDim.x) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;  y += blockDim.y * gridDim.y) {

            int idx = IDX2(x,y,height);
            Color3f val = flt[idx] / wgtsum[idx];

            image[idx] = val;
            val = val.toLinearRGB();
            surf2Dwrite(make_float4(val.x(),val.y(),val.z(),1.0f), 
                        cuda_data, (int) sizeof(float4) * x, height - y,
                        cudaBoundaryModeClamp);
        }
    }
}



// see lecture 16a-slide 16
__global__ void nl_means_filter_gpu(Color3f* image, float* wgtsum, Color3f* flt, int width, int height, Scene* scene)
{
    const int r = scene->filter_r;
    const int f = scene->filter_f;
    const float sigma = scene->filter_sigma;
    const float sigma2 = sigma*sigma;
    //const float h2=0.45f*0.45f;
    const float h = scene->filter_h;
    const float h2=h*h;

    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;  x += blockDim.x * gridDim.x) {
       for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;  y += blockDim.y * gridDim.y) {

            // loop over neighbors
            int minX = max(0,x-r);
            int maxX = min(width-1,x+r);
            int minY = max(0,y-r);
            int maxY = min(height-1,y+r);

            for (int i=minX; i<=maxX; ++i)
            {
                for (int j=minY; j<=maxY; ++j)
                {   
                    // here we consider neighbour (i,j)
                    float d2patch = 0;

                    int patchMinX = max(max(-f, -x), -i);
                    int patchMaxX = min(min(f, width-x-1), width-i-1);

                    int patchMinY = max(max(-f,-y), -j);
                    int patchMaxY = min(min(f, height-y-1), height-j-1);

                    // loop over patch of size (2f+1)^2 
                    for (int k=patchMinX;k<=patchMaxX;++k)
                    {
                        for (int l=patchMinY;l<=patchMaxY;++l)
                        {
                            Color3f color_p = image[IDX2(x+k,y+l,height)];
                            Color3f color_q = image[IDX2(i+k,j+l,height)];

                            float w = exp(-(k*k+l*l)/(2.0f*sigma2));
                            float patchv = (COL_NORMSQ((color_q-color_p).eval()))/h2;
                            d2patch += w*patchv;
                        }
                    }

                    d2patch /= ((patchMaxX-patchMinX+1)*(patchMaxY-patchMinY+1));

                    float wgt = exp(-max(d2patch,0.0f));
                    
                    int idx = IDX2(x,y,height);
                    wgtsum[idx] += wgt;
                    flt[idx] += wgt*image[IDX2(i,j,height)];
                }
            }
        }
    }
}

/**
 *
 * @param resource
 * @param w
 * @param h
 * @param scene  scene object on GPU memory
 */

static float filterInvoked = false;
static std::chrono::milliseconds filterStartTime;
static float* filterWeights;
static Color3f* filterOut;

void filter_scene(cudaSurfaceObject_t resource, int w, int h, nori::Scene *scene,Color3f *image)
{
    if (!filterInvoked)
    {
        int blockSize;
        int gridSize;

        cudaOccupancyMaxPotentialBlockSize(&gridSize,&blockSize,nl_means_filter_gpu,0,w*h);

        //we want to render 2D blocks not lines
        int blockW = sqrt(blockSize);
        int blockH = blockSize/blockW;
        dim3 block(blockW,blockH);
        int gridSizeW = (w + blockW - 1) / blockW;
        int gridSizeH = (h + blockH - 1) / blockH;
        dim3 grid(gridSizeW,gridSizeH);


        //cudaMalloc((void **) &filterImage, w * h * sizeof(Color3f));
        //cudaMemset(filterImage, 0, w * h * sizeof(Color3f));

        cudaMalloc((void **) &filterWeights, w * h * sizeof(float));
        cudaMemset(filterWeights, 0, w * h * sizeof(float));

        cudaMalloc((void **) &filterOut, w * h * sizeof(Color3f));
        cudaMemset(filterOut, 0, w * h * sizeof(Color3f));

        filterStartTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());


        // copy resource to image
        //std::cout<<"filter copy start"<<std::endl;
        //nori::copy_surface_to_data << < grid,block >> > (resource, w,h, image);
        //std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;

        // wait until finish
        cudaDeviceSynchronize();

        // actual filter
        std::cout<<"filter start"<<std::endl;
        nori::nl_means_filter_gpu << < grid,block >> > (image, filterWeights, filterOut, w, h, scene);
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;

        // wait until finish
        cudaDeviceSynchronize();

        std::cout<<"filter copy back start"<<std::endl;
        nori::copy_weighted_data_to_surface << < grid,block >> > (resource, image, w,h, filterOut, filterWeights);
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;

        filterInvoked = true;
    }

    if (cudaSuccess==cudaStreamQuery(0)){
        cudaDeviceSynchronize();

        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - filterStartTime;
        std::cout << "filter finished: Took" << float(diff.count())/1000  << " seconds!"<< std::endl ;
    }
}

NORI_NAMESPACE_END
