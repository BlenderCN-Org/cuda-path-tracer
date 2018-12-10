
#if !defined(_NORIGL_BUMP_MAP_H_)
#define _NORIGL_BUMP_MAP_H_

#include <nori/normalModifier.h>
#include <nori/frame.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

NORI_NAMESPACE_BEGIN

class BumpMap : public NormalModifier
{
public:

    __host__ BumpMap(const PropertyList &props);
    __host__ virtual std::string toString() const override;
    __host__ virtual void gpuTransfer(NoriObject **) override;
    __host__ void transferImage(unsigned char** target, cudaTextureObject_t* image_tex, cv::Mat& image);
    __host__ virtual size_t getSize() const override {return sizeof(BumpMap);};

    __device__ Vector3f eval(const Point2f& uv, const Frame& frame);

protected:

    Point2f m_delta;
    Vector2f m_scale;
    cv::Mat image; // image on cpu

    float height;
    float grad_delta;

    cudaTextureObject_t image_tex_x; // image on gpu
    cudaTextureObject_t image_tex_y; // image on gpu
    unsigned char* gpu_data_x;
    unsigned char* gpu_data_y;
    size_t gpu_step;
};


NORI_NAMESPACE_END

#endif /* _NORIGL_BUMP_MAP_H_ */
