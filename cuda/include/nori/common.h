/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__NORI_COMMON_H)
#define __NORI_COMMON_H

#if defined(_MSC_VER)
/* Disable some warnings on MSVC++ */
#pragma warning(disable : 4127 4702 4100 4515 4800 4146 4512)
#define WIN32_LEAN_AND_MEAN     /* Don't ever include MFC on Windows */
#define NOMINMAX                /* Don't override min/max */
#endif
#ifdef __JETBRAINS_IDE__
    #define __CUDACC__ 1
    #define __host__
    #define __device__
    #define __global__
    #define __forceinline__
    #define __shared__
    inline void __syncthreads() {}
    inline void __threadfence_block() {}
    template<class T> inline T __clz(const T val) { return val; }
    struct __cuda_fake_struct { int x; };
    extern __cuda_fake_struct blockDim;
    extern __cuda_fake_struct threadIdx;
    extern __cuda_fake_struct blockIdx;
#endif
/* Include the basics needed by any Nori file */
#define EIGEN_USE_GPU
#ifndef __CUDA_ARCH__
#include <iostream>
#endif

#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <stdint.h>

#include <host_defines.h>
//#include <ImathPlatform.h>
#include <tinyformat.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <texture_indirect_functions.h>
/* Convenience definitions */
#define NORI_NAMESPACE_BEGIN namespace nori {
#define NORI_NAMESPACE_END }

#if defined(__NORI_APPLE__NORI_)
#define PLATFORM_MACOS
#elif defined(__NORI_linux__NORI_)
#define PLATFORM_LINUX
#elif defined(WIN32)
#define PLATFORM_WINDOWS
#endif
#define stdmin(x,y) ((x<y)?x:y)
#define stdmax(x,y) ((y>x)?y:x)

/* "Ray epsilon": relative error threshold for ray intersection computations */
#define Epsilon 1e-4f

/* A few useful constants */
#undef M_PI

#define M_PI         3.14159265358979323846f
#define INV_PI       0.31830988618379067154f
#define INV_TWOPI    0.15915494309189533577f
#define INV_FOURPI   0.07957747154594766788f
#define SQRT_TWO     1.41421356237309504880f
#define INV_SQRT_TWO 0.70710678118654752440f

/* Forward declarations */
namespace filesystem {
    class path;
    class resolver;
};

#ifdef __CUDACC__
#define nullptr NULL
#endif

#define CHECK_ERROR( err ) \
  if( err != cudaSuccess ) { \
    std::cerr << "CUDA ERROR: " << cudaGetErrorString( err ) << std::endl; \
    exit( -1 ); \
  }


NORI_NAMESPACE_BEGIN
typedef uint64_t Sampler ;
/* Forward declarations */
    template <typename Scalar, int Dimension>  struct TVector;
    template <typename Scalar, int Dimension>  struct TPoint;
    template <typename Point, typename Vector> struct TRay;
    template <typename Point>                  struct TBoundingBox;

/* Basic Nori data structures (vectors, points, rays, bounding boxes,
   kd-trees) are oblivious to the underlying data type and dimension.
   The following list of typedefs establishes some convenient aliases
   for specific types. */
    typedef TVector<float, 1>       Vector1f;
    typedef TVector<float, 2>       Vector2f;
    typedef TVector<float, 3>       Vector3f;
    typedef TVector<float, 4>       Vector4f;
    typedef TVector<double, 1>      Vector1d;
    typedef TVector<double, 2>      Vector2d;
    typedef TVector<double, 3>      Vector3d;
    typedef TVector<double, 4>      Vector4d;
    typedef TVector<int, 1>         Vector1i;
    typedef TVector<int, 2>         Vector2i;
    typedef TVector<int, 3>         Vector3i;
    typedef TVector<int, 4>         Vector4i;
    typedef TPoint<float, 1>        Point1f;
    typedef TPoint<float, 2>        Point2f;
    typedef TPoint<float, 3>        Point3f;
    typedef TPoint<float, 4>        Point4f;
    typedef TPoint<double, 1>       Point1d;
    typedef TPoint<double, 2>       Point2d;
    typedef TPoint<double, 3>       Point3d;
    typedef TPoint<double, 4>       Point4d;
    typedef TPoint<int, 1>          Point1i;
    typedef TPoint<int, 2>          Point2i;
    typedef TPoint<int, 3>          Point3i;
    typedef TPoint<int, 4>          Point4i;
    typedef TBoundingBox<Point1f>   BoundingBox1f;
    typedef TBoundingBox<Point2f>   BoundingBox2f;
    typedef TBoundingBox<Point3f>   BoundingBox3f;
    typedef TBoundingBox<Point4f>   BoundingBox4f;
    typedef TBoundingBox<Point1d>   BoundingBox1d;
    typedef TBoundingBox<Point2d>   BoundingBox2d;
    typedef TBoundingBox<Point3d>   BoundingBox3d;
    typedef TBoundingBox<Point4d>   BoundingBox4d;
    typedef TBoundingBox<Point1i>   BoundingBox1i;
    typedef TBoundingBox<Point2i>   BoundingBox2i;
    typedef TBoundingBox<Point3i>   BoundingBox3i;
    typedef TBoundingBox<Point4i>   BoundingBox4i;
    typedef TRay<Point2f, Vector2f> Ray2f;
    typedef TRay<Point3f, Vector3f> Ray3f;

/// Some more forward declarations
    class BSDF;
    class Bitmap;
    class BlockGenerator;
    class Camera;
    class ImageBlock;
    class Integrator;
    class KDTree;
    class Emitter;
    struct EmitterQueryRecord;
    class Shape;
    class NoriObject;
    class NoriObjectFactory;
    class NoriScreen;
    class PhaseFunction;
    class ReconstructionFilter;
    class Scene;

/// Import cout, cerr, endl for debugging purposes
    using std::cout;
    using std::cerr;
    using std::endl;

    typedef Eigen::Matrix<float,    Eigen::Dynamic, Eigen::Dynamic> MatrixXf;
    typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXu;

/// Simple exception class, which stores a human-readable error description
    class NoriException : public std::runtime_error {
    public:
        /// Variadic template constructor to support printf-style arguments
        template <typename... Args> NoriException(const char *fmt, const Args &... args)
                : std::runtime_error(tfm::format(fmt, args...)) { }
    };

/// Return the number of cores (real and virtual)
    extern int getCoreCount();

/// Indent a string by the specified number of spaces
    extern std::string indent(const std::string &string, int amount = 2);

/// Convert a string to lower case
    extern std::string toLower(const std::string &value);

/// Convert a string into an boolean value
    extern bool toBool(const std::string &str);

/// Convert a string into a signed integer value
    extern int toInt(const std::string &str);

/// Convert a string into an unsigned integer value
    extern unsigned int toUInt(const std::string &str);

/// Convert a string into a floating point value
    extern float toFloat(const std::string &str);

/// Check token size to distinguish between vector2 and vector3
    extern size_t vectorSize(const std::string &str);

/// Convert a string into a 3D vector
    extern Eigen::Vector2f toVector2f(const std::string &str);
/// Convert a string into a 3D vector
    extern Eigen::Vector3f toVector3f(const std::string &str);

/// Tokenize a string into a list by splitting at 'delim'
    extern std::vector<std::string> tokenize(const std::string &s, const std::string &delim = ", ", bool includeEmpty = false);

/// Check if a string ends with another string
    extern bool endsWith(const std::string &value, const std::string &ending);

/// Convert a time value in milliseconds into a human-readable string
    extern std::string timeString(double time, bool precise = false);

/// Convert a memory amount in bytes into a human-readable string
    extern std::string memString(size_t size, bool precise = false);

/// Measures associated with probability distributions
    enum EMeasure {
        EUnknownMeasure = 0,
        ESolidAngle,
        EDiscrete
    };

//// Convert radians to degrees
    __host__ __device__  inline float radToDeg(float value) { return value * (180.0f / M_PI); }

/// Convert degrees to radians
    __host__ __device__  inline float degToRad(float value) { return value * (M_PI / 180.0f); }

#if !defined(_GNU_SOURCE)
    /// Emulate sincosf using sinf() and cosf()
    inline void sincosf(float theta, float *_sin, float *_cos) {
        *_sin = sinf(theta);
        *_cos = cosf(theta);
    }
#endif

/// Simple floating point clamping function
    __host__ __device__ inline float clamp(float value, float min, float max) {
        if (value < min)
            return min;
        else if (value > max)
            return max;
        else return value;
    }

/// Simple integer clamping function
    __host__ __device__ inline int clamp(int value, int min, int max) {
        if (value < min)
            return min;
        else if (value > max)
            return max;
        else return value;
    }

/// Linearly interpolate between two values
    __host__ __device__ inline float lerp(float t, float v1, float v2) {
        return ((float) 1 - t) * v1 + t * v2;
    }

/// Always-positive modulo operation
    __host__ __device__ inline int mod(int a, int b) {
        int r = a % b;
        return (r < 0) ? r+b : r;
    }

/// Compute a direction for the given coordinates in spherical coordinates
    __device__ extern Vector3f sphericalDirection(float theta, float phi);

/// Compute a direction for the given coordinates in spherical coordinates
    __device__  extern Point2f sphericalCoordinates(const Vector3f &dir);

/**
 * \brief Calculates the unpolarized fresnel reflection coefficient for a
 * dielectric material. Handles incidence from either side (i.e.
 * \code cosThetaI<0 is allowed).
 *
 * \param cosThetaI
 *      Cosine of the angle between the normal and the incident ray
 * \param extIOR
 *      Refractive index of the side that contains the surface normal
 * \param intIOR
 *      Refractive index of the interior
 */
    __device__ extern float fresnel(float cosThetaI, float extIOR, float intIOR);

/**
 * \brief Return the global file resolver instance
 *
 * This class is used to locate resource files (e.g. mesh or
 * texture files) referenced by a scene being loaded
 */
    extern filesystem::resolver *getFileResolver();

/**
 *
 * This class implements some vector methods, (and internally uses a vector on the cpu side) such that we can create it on cpu and move to gpu
 * On gpu we have less operators availible, as resizing e.g. is not possible
 * @tparam T
 */

    template <class T>
    class CombinedVector{
    public:
#ifdef __CUDA_ARCH__
        __device__ T &operator[](size_t i ) {
                    printf("wriiiite\n");
                    return m_gpuData[i];
            }
            __device__ T operator[](size_t i ) const{
                    T *ret;

                    switch(sizeof(T)){
                       case(4):{
                                int v = tex1Dfetch<int>(tex,i);
                                T t = *(T*)&v;
                                return t;
                            }
                       case(8):{
                                int2 v = tex1Dfetch<int2>(tex,i);
                                 T t = *(T*)&v;
                                return t;
                       }
                       case(32):{
                                //long4 test;
                                //tex1Dfetch<long4>(&test,tex,i);
                                struct {
                                    int4 v1;
                                    int4 v2;
                                } v;
                                v.v1 = tex1Dfetch<int4>(tex,2*i);
                                v.v2 = tex1Dfetch<int4>(tex,2*i+1);
                                T t = *(T*)&v;
                                return t;


                       }

                    }
                    return m_gpuData[i];
            }


            __device__ size_t size() const{
                    return sizeGpu;
            }
            __device__ const T* begin() const{
                return m_gpuData;
            }
             __device__  T* begin() {
                return m_gpuData;
            }

            __device__ T* end() {
                return m_gpuData + sizeGpu;
            }
            __device__ const T* end() const{
                return m_gpuData + sizeGpu;
            }
           __device__ T& back() {
                 return m_gpuData[sizeGpu-1];
           }
         __device__ const T& back() const{
                 return m_gpuData[sizeGpu-1];
           }
        void operator= (std::vector<T> a){
            m_data = a;
        }
#else
        const T &operator[](size_t i ) const {
            return m_data[i];
        }
        T &operator[](size_t i ) {
            return m_data[i];
        }
        size_t size() const{
            return m_data.size();
        }


        T* data() {
            return m_data.data();
        }
        const T* data() const {
            return m_data.data();
        }
        T* begin() {
            return data();
        }
        const T* begin() const {
            return data();
        }
        T* end() {

            return begin()+m_data.size();
        }

        const T* end() const{
            return begin()+m_data.size();
        }
        T &back() {
            return m_data.back();
        }
        const T &back() const{
            return m_data.back();
        }
        void operator= (std::vector<T> a){
            m_data = a;
        }
#endif
        //host only functions
        void push_back(T elem){
            m_data.push_back(elem);
        }
        void clear(){
            m_data.clear();
        }
        void shrink_to_fit(){
            m_data.shrink_to_fit();
        }
        void resize(size_t n){
            m_data.resize(n);
        }

        void reserve(size_t i) {
            m_data.reserve(i);
        }
        //hack do not use unintended loads the first 8 bytes of BVHNode which includes everything but the bounding box
        __device__ const T getBVHfirst(size_t i) const{
            struct {
                int2 v1;
                int2 v2;
                int4 v3;
            } v;
            //TODO: move this to a cpp.cu file because gcc somehow can't cope with the call
            #ifdef __CUDA_ARCH__
                v.v1  = tex1Dfetch<int2>(tex,2*i);
            #endif
            T t = *(T*)&v;
            return t;
        }


        /**
         * Transfer the data from cpu to gpu
         * @return
         */
        __host__ void transferGpu(){

            sizeGpu = m_data.size();
            void **p = (void **) &m_gpuData;

            cudaMalloc(p,sizeof(T)*sizeGpu);

            cudaMemcpy(m_gpuData,m_data.data(),sizeof(T)*sizeGpu,cudaMemcpyHostToDevice);
            //for three cases 4 byte types 8 byte types and 32 byte types (last one because of BVHNode) we use texture memory
            if(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 32){
                cudaResourceDesc resDesc;
                memset(&resDesc, 0, sizeof(resDesc));
                resDesc.resType = cudaResourceTypeLinear;
                resDesc.res.linear.devPtr = m_gpuData;
                resDesc.res.linear.desc.f = cudaChannelFormatKind::cudaChannelFormatKindUnsigned;
                if(sizeof(T)==4) {
                    resDesc.res.linear.desc.x = 32; // bits per channel
                    resDesc.res.linear.desc.y = 0; // bits per channel
                    resDesc.res.linear.desc.z = 0; // bits per channel
                    resDesc.res.linear.desc.w = 0; // bits per channel
                }else if(sizeof(T)==8)  {
                    resDesc.res.linear.desc.x = 32; // bits per channel
                    resDesc.res.linear.desc.y = 32; // bits per channel
                    resDesc.res.linear.desc.z = 0; // bits per channel
                    resDesc.res.linear.desc.w = 0; // bits per channel
                }else if(sizeof(T)==32) {

                    resDesc.res.linear.desc.x = 32; // bits per channel
                    resDesc.res.linear.desc.y = 32; // bits per channel
                    resDesc.res.linear.desc.z = 32; // bits per channel
                    resDesc.res.linear.desc.w = 32; // bits per channel

                }

                resDesc.res.linear.sizeInBytes = sizeGpu * sizeof(T);

                cudaTextureDesc texDesc;
                memset(&texDesc, 0, sizeof(texDesc));

                texDesc.readMode = cudaReadModeElementType;
                //texDesc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
                cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

            }

            cudaMemcpy(m_gpuData,m_data.data(),sizeof(T)*sizeGpu,cudaMemcpyHostToDevice);
        }



    private:
        // const char* TypeName = typeid(T).name();
        cudaTextureObject_t tex;
        T* m_gpuData;
        size_t sizeGpu;
        std::vector<T> m_data;

    };

/**
 * CallCudaN to be extended, decides which function to call based on a type
 * N is the number of arguments as recursion is not possible for macros (and it would be ambigous anyway)
 */

/**
 * @param type Object class e.g. BSDF
 * @param field enum field in type which determines the subtype for the cast e.g. bsdfType
 * @param object object from which the function should be called
 * @param function the function which should be called e.g. sample
 * @param constant either const if object is const or nonconst (which is a empty define)
 * @param EN Enum for the n-th subclass      e.g. EDiffuse
 * @param TN classname for the n-th subclass e.g. Diffuse
 * @param __VA_ARGS__ arguments for the function
 */
#define nonconst
#define CallCuda1(type,field,object,function,constant,E1,T1,...) (  \
    (static_cast<constant T1*>(object))->function(__VA_ARGS__)\
    )

#define CallCuda2(type,field,object,function,constant,E1,T1,E2,T2,...) ( \
     (object->field == type::E1)?(static_cast<constant T1*>(object))->function(__VA_ARGS__):  \
     CallCuda1(type,field,object,function,constant,E2,T2,__VA_ARGS__) \
     )

#define CallCuda3(type,field,object,function,constant,E1,T1,E2,T2,E3,T3,...) ( \
    (object->field == type::E1)?(static_cast<constant T1*>(object))->function(__VA_ARGS__):  \
    CallCuda2(type,field,object,function,constant,E2,T2,E3,T3,__VA_ARGS__) \
    )

NORI_NAMESPACE_END


//#include <nori/shape.h>
//Eigen::MatrixXf unMap(Eigen::Map<Eigen::MatrixXf> arg);


#endif /* __NORI_COMMON_H */
