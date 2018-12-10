
#if !defined(_NORI_GL_NORMAL_MAP_)
#define _NORI_GL_NORMAL_MAP_

#include <nori/normalModifier.h>
#include <nori/frame.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

NORI_NAMESPACE_BEGIN

class NormalMap : public NormalModifier
{
public:

    __host__ NormalMap(const PropertyList &props);
    __host__ virtual std::string toString() const override;
    __host__ virtual void gpuTransfer(NoriObject **) override;
    __host__ void transferImage(unsigned char** target, cudaTextureObject_t* image_tex, cv::Mat& image);
    __host__ virtual size_t getSize() const override {return sizeof(NormalMap);};

    __device__ Vector3f eval(const Point2f& uv, const Frame& frame);

protected:

    Point2f m_delta;
    Vector2f m_scale;
    cv::Mat image; // image on cpu

    cudaTextureObject_t image_tex; // image on gpu
    unsigned char* gpu_data;
    size_t gpu_step;
};


NORI_NAMESPACE_END

#endif /* _NORI_GL_NORMAL_MAP_ */
